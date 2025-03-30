import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from torch.nn.init import xavier_uniform_, constant_
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule, build_norm_layer, build_activation_layer
from ..decode_head import BaseDecodeHead
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmengine.logging import print_log
from mmengine.model import BaseModule, ModuleList
from mmengine.registry import MODELS
from mmseg.utils import SampleList
from timm.models.layers import trunc_normal_, DropPath
from ...utils import resize
from ...losses import accuracy
from ..segmenter_mask_head import SegmenterMaskTransformerHead

from ops_dcnv3.functions import DCNv3Function, dcnv3_core_pytorch
from ops_dcnv3.modules import DCNv3_pytorch
from ops_dcnv3 import modules as opsm

class ConvBNGeLU(nn.Module):
    def __init__(self, 
                 in_channels,
                 channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=False,
                 inplace=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, channels, kernel_size, stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.gelu(x)
        return x

class DCNv3(nn.Module):
    def __init__(
            self,
            channels=64,
            kernel_size=3,
            stride=1,
            pad=1,
            dilation=1,
            group=4,
            offset_scale=1.0,
            act_layer='GELU',
            norm_layer='LN',
            center_feature_scale=False):

        super().__init__()

        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale

        self.offset = nn.Linear(channels, group * kernel_size * kernel_size * 2)
        self.mask = nn.Linear(channels, group * kernel_size * kernel_size)
        self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)
        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)
        constant_(self.mask.weight.data, 0.)
        constant_(self.mask.bias.data, 0.)

    def forward(self, input):
        """
        :param query                       (N, H, W, C)
        :return output                     (N, H, W, C)
        """
        b, c, h, w = input.shape
        x = input.view(b, h, w, c)

        #x1 = offset.view(b, h, w, c)
        offset = self.offset(x)
        mask = self.mask(x).reshape(b, h, w, self.group, -1)
        mask = F.softmax(mask, -1).reshape(b, h, w, -1)

        x = dcnv3_core_pytorch(
            x, offset, mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale)

        x = x.view(b, c, h, w).contiguous()

        return x

class DCNBNGeLU(nn.Module):
    def __init__(self, 
                 in_channels,
                 channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=False,
                 inplace=True):
        super().__init__()
        self.conv1 = ConvBNGeLU(in_channels, channels)
        self.conv2 = DCNv3(channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.gelu(x)
        return x

class PostTransformerEncoderLayer(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 batch_first=True,
                 attn_cfg=dict(),
                 ffn_cfg=dict(),
                 with_cp=False):
        super().__init__()

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        attn_cfg.update(
            dict(
                embed_dims=embed_dims,
                num_heads=num_heads,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
                batch_first=batch_first,
                bias=qkv_bias))

        self.build_attn(attn_cfg)

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        ffn_cfg.update(
            dict(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                num_fcs=num_fcs,
                ffn_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate)
                if drop_path_rate > 0 else None,
                act_cfg=act_cfg))
        self.build_ffn(ffn_cfg)
        self.with_cp = with_cp

    def build_attn(self, attn_cfg):
        self.attn = MultiheadAttention(**attn_cfg)

    def build_ffn(self, ffn_cfg):
        self.ffn = FFN(**ffn_cfg)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):

        def _inner_forward(x):
            
            x = self.norm1(self.attn(x)) + x
            #identity=x
            #x = self.attn(x)
            #x = self.norm1(x) + identity

            x = self.norm2(self.ffn(x)) + x
            #identity=x
            #x = self.ffn(x)
            #x = self.norm2(x) + identity
            #x = self.attn(self.norm1(x), identity=x)
            #x = self.ffn(self.norm2(x), identity=x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x

class AuxHead(SegmenterMaskTransformerHead):
    def __init__(self, **kwargs):
        super(AuxHead, self).__init__(**kwargs)

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.num_layers)]
        self.layers = ModuleList()
        for i in range(self.num_layers):
            self.layers.append(
                PostTransformerEncoderLayer(
                    embed_dims=self.embed_dims,
                    num_heads=self.num_heads,
                    feedforward_channels=self.mlp_ratio * self.embed_dims,
                    attn_drop_rate=self.attn_drop_rate,
                    drop_rate=self.drop_rate,
                    drop_path_rate=dpr[i],
                    num_fcs=self.num_fcs,
                    qkv_bias=self.qkv_bias,
                    act_cfg=dict(type='GELU'),
                    norm_cfg=dict(type='LN'),
                    batch_first=True,
                ))
        
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(b, -1, c)
        x = self.dec_proj(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for layer in self.layers:
            x = layer(x)
        x = self.decoder_norm(x)

        patches = self.patch_proj(x[:, :-self.num_classes])
        cls_seg_feat = self.classes_proj(x[:, -self.num_classes:])

        patches = F.normalize(patches, dim=2, p=2)
        cls_seg_feat = F.normalize(cls_seg_feat, dim=2, p=2)

        
        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks)
        masks = masks.permute(0, 2, 1).contiguous().view(b, -1, h, w)
        return masks, cls_seg_feat

class ChannelMixAttentionModule(nn.Module):
    def __init__(self,
                 in_channels,
                 channels,
                 kernel_size=[3, 5, 7]):
        super().__init__()

        self.split_channels = channels // len(kernel_size)

        self.pre = nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.post = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.ModuleList()
        self.kernel_size = kernel_size

        for k in kernel_size:
            self.conv.append(ConvBNGeLU(self.split_channels, self.split_channels, kernel_size=k))
    def forward(self, x):
        x = self.pre(x)

        outs = []
        #将输入x分成len(kernel_size)份，每份self.split_channels个通道
        temp = torch.chunk(x, len(self.kernel_size), dim=1)
        for i in range(len(self.kernel_size)):
            outs.append(self.conv[i](temp[i]))
        
        outs = torch.cat(outs, dim=1)
        outs = self.post(outs)

        return outs

class MSCAAttention(BaseModule):
    def __init__(self,
                 channels,
                 kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 paddings=[2, [0, 3], [0, 5], [0, 10]]):
        super().__init__()
        self.conv0 = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_sizes[0],
            padding=paddings[0],
            groups=channels)
        for i, (kernel_size,
                padding) in enumerate(zip(kernel_sizes[1:], paddings[1:])):
            kernel_size_ = [kernel_size, kernel_size[::-1]]
            padding_ = [padding, padding[::-1]]
            conv_name = [f'conv{i}_1', f'conv{i}_2']
            for i_kernel, i_pad, i_conv in zip(kernel_size_, padding_,
                                               conv_name):
                self.add_module(
                    i_conv,
                    nn.Conv2d(
                        channels,
                        channels,
                        tuple(i_kernel),
                        padding=i_pad,
                        groups=channels))
        self.conv3 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        u = x.clone()

        attn = self.conv0(x)

        # Multi-Scale Feature extraction
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)

        x = attn * u

        return x

class MSCASpatialAttention(BaseModule):
    def __init__(self,
                 in_channels,
                 attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
                 act_cfg=dict(type='GELU')):
        super().__init__()
        self.proj_1 = nn.Conv2d(in_channels, in_channels, 1)
        self.activation = build_activation_layer(act_cfg)
        self.spatial_gating_unit = MSCAAttention(in_channels,
                                                 attention_kernel_sizes,
                                                 attention_kernel_paddings)
        self.proj_2 = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        shorcut = x.clone()

        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        
        x = x + shorcut

        return x

class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=8):
        super(ChannelAttention, self).__init__()

        self.channels = channels

        self.weight = MSCASpatialAttention(channels)
        self.softmax = nn.Softmax(dim=2)

        self.se = nn.Sequential(
            nn.Conv2d(channels, channels // ratio, kernel_size=1),
            nn.LayerNorm([channels // ratio, 1, 1]),
            nn.GELU(),
            nn.Conv2d(channels // ratio, channels, kernel_size=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        weight = self.weight(x)
        weight = weight.view(B, C, H*W)
        weight = self.softmax(weight)
        weight = weight.view(B, C, H, W)
        attn = x * weight
        attn = attn.view(B, C, H*W)
        attn = torch.sum(attn, dim=2)
        attn = attn.view(B, C, 1, 1)
        attn = self.se(attn)
        return attn

class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super(SpatialAttention, self).__init__()

        self.channels = channels

        self.pool = nn.AdaptiveAvgPool2d(2)

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        self.softmax = nn.Softmax(dim=1)

        self.conv2 = nn.Sequential(
            MSCASpatialAttention(4),
            nn.BatchNorm2d(4),
            nn.GELU(),
            nn.Conv2d(4, 1, kernel_size=1)
        )
    
    def forward(self, x):
        B, C, H, W = x.shape

        weight = self.pool(x)
        weight = self.conv1(weight)
        weight = self.softmax(weight)
        weight = weight.view(B, C, 1, 1, -1)

        x = x.view(B, C, H, W, 1)
        attn = x * weight
        attn = torch.sum(attn, dim=1)
        attn = attn.permute(0, 3, 1, 2)
        out = self.conv2(attn)
        return out

class MultiChannelsAlignment(nn.Module):
    def __init__(self,
                channels):
        super().__init__()

        self.cam = ChannelAttention(channels)
        self.sam = SpatialAttention(channels)

        self.fusion1 = ConvBNGeLU(channels, channels, kernel_size=3, groups=channels)
        self.fusion2 = ConvBNGeLU(channels, channels, kernel_size=3, groups=channels)
    def forward(self, x):
        attn_cam = self.cam(x)
        out = self.fusion1(x + attn_cam)

        attn_sam = self.sam(out)
        out = self.fusion2(out + attn_sam)

        return out

@MODELS.register_module()
class Custom9(BaseDecodeHead):
    def __init__(self,
                norm_cfg,
                 **kwargs):
        super(Custom9, self).__init__(**kwargs)

        self.scale = 1

        self.input_transform = 'multiple_select'

        self.up4 = DCNBNGeLU(self.in_channels[3] + self.in_channels[3], self.in_channels[3])
        self.attn4 = MultiChannelsAlignment(self.in_channels[3])

        self.up3 = DCNBNGeLU(self.in_channels[2] + self.in_channels[3], self.in_channels[2])
        self.attn3 = MultiChannelsAlignment(self.in_channels[2])

        self.up2 = DCNBNGeLU(self.in_channels[1] + self.in_channels[2], self.in_channels[1])
        self.attn2 = MultiChannelsAlignment(self.in_channels[1])

        self.up1 = DCNBNGeLU(self.in_channels[0] + self.in_channels[1], self.channels)
        self.attn1 = MultiChannelsAlignment(self.in_channels[0])

        self.aux_head = AuxHead(
            in_channels=self.in_channels[0],
            channels=self.channels,
            num_layers=2,
            num_classes=self.num_classes,
            num_heads=12,
            embed_dims=self.channels,
            dropout_ratio=0.1)
        #self.conv = nn.Conv2d(self.in_channels[3] + self.in_channels[2] + self.in_channels[1], self.in_channels[0], 1, 1, 0)

        self.cmm = ChannelMixAttentionModule(self.in_channels[3] + self.in_channels[2] + self.in_channels[1], self.in_channels[0])

    def _forward_feature(self, inputs):
        f1, f2, f3, f4 = self._transform_inputs(inputs)

        aux = []

        a4 = self.attn4(f4)
        up4 = torch.cat([a4, f4], dim=1)
        up4 = self.up4(up4)
        aux.append(up4)

        up4 = F.interpolate(up4, size=f3.shape[2:], mode='bilinear')
        a3 = self.attn3(f3)
        up3 = torch.cat([a3, up4], dim=1)
        up3 = self.up3(up3)
        aux.append(up3)

        up3 = F.interpolate(up3, size=f2.shape[2:], mode='bilinear')
        a2 = self.attn2(f2)
        up2 = torch.cat([a2, up3], dim=1)
        up2 = self.up2(up2)
        aux.append(up2)

        up2 = F.interpolate(up2, size=f1.shape[2:], mode='bilinear')
        a1 = self.attn1(f1)
        up1 = torch.cat([a1, up2], dim=1)
        up1 = self.up1(up1)

        main_out = up1

        b, c, h, w = main_out.shape

        for i in range(len(aux)):
            aux[i] = F.interpolate(aux[i], size=f3.shape[2:], mode='bilinear')
        aux_feats = torch.cat(aux, dim=1)
        #aux_feats = self.conv(aux_feats)
        aux_feats = self.cmm(aux_feats)
        
        aux_probs, cls_emb = self.aux_head(aux_feats)
        
        main_out = main_out.permute(0, 2, 3, 1).contiguous().view(b, -1, c)
        main_out = main_out @ cls_emb.transpose(1, 2)
        aux_probs = F.interpolate(aux_probs, size=up1.shape[2:], mode='bilinear')
        main_out = main_out.permute(0, 2, 1).contiguous().view(b, -1, h, w)

        return main_out, aux_probs
    
    def forward(self, inputs):
        feats, probs = self._forward_feature(inputs)
        #x = self.cls_seg(feats)
        return feats, probs
    
    def loss(self, inputs, batch_data_samples, train_cfg):
        seg_logits, probs = self.forward(inputs)
        logits = [seg_logits, probs]
        return self.loss_by_feat(logits, batch_data_samples)
    
    def loss_by_feat(self, seg_logits, batch_data_samples):
        loss = dict()
        seg_label = self._stack_batch_gt(batch_data_samples)
        seg_logits[0] = resize(input=seg_logits[0], size=seg_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
        seg_logits[1] = resize(input=seg_logits[1], size=seg_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits[0],
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index) + 0.4*loss_decode(
                    seg_logits[1],
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logits[0],
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index) + 0.4*loss_decode(
                    seg_logits[1],
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)

        loss['acc_seg'] = accuracy(
            seg_logits[0], seg_label, ignore_index=self.ignore_index)
        return loss

    def predict(self, inputs, batch_img_metas, test_cfg):
        seg_logits, probs = self.forward(inputs)
        return self.predict_by_feat(seg_logits, batch_img_metas)