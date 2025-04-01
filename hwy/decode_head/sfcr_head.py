# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from torchvision.ops import DeformConv2d

from mmseg.ops import resize
from timm.models.layers import DropPath
from ...builder import HEADS
from ..decode_head import BaseDecodeHead


class ConvBNGELU(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False, groups=1):
        super(ConvBNGELU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2, groups=groups),
            norm_layer(out_channels),
            nn.GELU()
        )


class ConvBN(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=True):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class DCNv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super().__init__()
        self.dcn = DeformConv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        self.offset_mask = nn.Conv2d(out_channels,  groups* 3 * kernel_size * kernel_size, kernel_size, stride, padding)
        self._init_offset()

    def _init_offset(self):
        self.offset_mask.weight.data.zero_()
        self.offset_mask.bias.data.zero_()

    def forward(self, x, offset):
        out = self.offset_mask(offset)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat([o1, o2], dim=1)
        mask = mask.sigmoid()
        return self.dcn(x, offset, mask)


class ReparamLargeKernelConvBNGELU(nn.Module):

    def __init__(self, in_channels, out_channels, large_kernel_size=31, stride=1, groups=1, small_kernel_size=3):
        super(ReparamLargeKernelConvBNGELU, self).__init__()
        self.large_kernel_size = large_kernel_size
        self.small_kernel_size = small_kernel_size
        self.small_kernel_conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=small_kernel_size, 
            padding=((stride - 1) + (small_kernel_size - 1)) // 2,
            groups=groups,
            stride=stride,
            dilation=1,
            bias=False)
        self.bn_small = nn.BatchNorm2d(out_channels)
        self.large_kernel_conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=large_kernel_size,
            padding=((stride - 1) + (large_kernel_size - 1)) // 2,
            groups=groups,
            stride=stride,
            dilation=1,
            bias=False
        )
        self.bn_large = nn.BatchNorm2d(out_channels)
        self.reparam_conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=large_kernel_size,
            padding=((stride - 1) + (large_kernel_size - 1)) // 2,
            groups=groups,
            stride=stride,
            dilation=1,
            bias=True
        )
        self.act = nn.GELU()
        self.need_reparam = True

    def forward(self, x):
        if self.training:
            if not self.need_reparam:
                self.need_reparam = True
            out = self.train_forward(x)
        else:
            if self.need_reparam:
                self.reparam()
                self.need_reparam = False
            out = self.test_forward(x)
        return out

    def train_forward(self, x):
        out = self.bn_large(self.large_kernel_conv(x)) + self.bn_small(self.small_kernel_conv(x))
        out = self.act(out)
        return out

    def test_forward(self, x):
        out = self.reparam_conv(x)
        out = self.act(out)
        return out

    def reparam(self):
        eq_k, eq_b = self.fuse_bn(self.large_kernel_conv, self.bn_large)
        small_k, small_b = self.fuse_bn(self.small_kernel_conv, self.bn_small)
        eq_b += small_b
        eq_k += F.pad(small_k, [(self.large_kernel_size - self.small_kernel_size) // 2] * 4)
        self.reparam_conv.weight.data = eq_k
        self.reparam_conv.bias.data = eq_b

    def fuse_bn(self, conv, bn):
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class FeatureInteractionProcess(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.dcn_high = DCNv2(channels, channels, kernel_size=3, stride=1, padding=1)
        self.dcn_low = DCNv2(channels, channels, kernel_size=3, stride=1, padding=1)
        self.offset = nn.Conv2d(channels*2, channels, kernel_size=1, bias=False)

    def forward(self, high, low):
        high = F.interpolate(high, size=low.shape[2:], mode='bilinear', align_corners=True)
        offset = torch.cat([high, low], dim=1)
        offset = self.offset(offset)
        high = self.dcn_high(high, offset)
        low = self.dcn_low(low, offset)
        return high, low


class ComplementaryFusion(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fusion = ConvBNGELU(channels, channels, kernel_size=3, groups=channels)
    
    def forward(self, high, low):
        B, C, H, W = high.shape
        attn1 = self.softmax(high)
        attn1 = attn1.permute(0, 2, 3, 1).view(B, H*W, C)
        attn1 = self.pool(attn1)
        attn1 = attn1.view(B, 1, H, W)
        attn2 = self.softmax(low)
        attn2 = attn2.permute(0, 2, 3, 1).view(B, H*W, C) 
        attn2 = self.pool(attn2)
        attn2 = attn2.view(B, 1, H, W)
        high = high * (1 - attn2)
        low = low * (1 - attn1)
        out = self.fusion(high + low)
        return out
    

class StairFusionNetworkBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.feature_process = FeatureInteractionProcess(channels)
        self.fusion = ComplementaryFusion(channels)
    
    def forward(self, high, low):
        high, low = self.feature_process(high, low)
        out = self.fusion(high, low)
        return out


class StairFusionNetworkPath(nn.Module):

    def __init__(self, channels, depth):
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.block_list = nn.ModuleList()
        for _ in range(depth):
            block = StairFusionNetworkBlock(channels)
            self.block_list.append(block)

    def forward(self, x):
        out = []
        temp = x[0]
        for i in range(self.depth):
            temp = self.block_list[i](temp, x[i + 1])
            out.append(temp)
        return out


class StairFusionNetwork(nn.Module):

    def __init__(self, channels, depth):
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.sfn_path_list = nn.ModuleList()
        for i in range(depth, 0, -1):
            path = StairFusionNetworkPath(channels, i)
            self.sfn_path_list.append(path)

    def forward(self, x):
        out = x[::-1]
        for i in range(self.depth):
            out = self.sfn_path_list[i](out)
        return out[0]


class ChannelAttention(nn.Module):

    def __init__(self, channels, reduce_ratio=8, expand=4, kernel_size=31):
        super(ChannelAttention, self).__init__()
        self.channels = channels
        self.weight = nn.Sequential(
            ReparamLargeKernelConvBNGELU(channels, channels, large_kernel_size=kernel_size, groups=channels),
            Conv(channels, channels, kernel_size=1)
        )
        self.softmax = nn.Softmax(dim=2)
        self.se = nn.Sequential(
            nn.Conv2d(channels, channels//reduce_ratio, kernel_size=1),
            nn.LayerNorm([channels//reduce_ratio, 1, 1]),
            nn.GELU(),
            nn.Conv2d(channels//reduce_ratio, channels, kernel_size=1)
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
        attn_se = self.se(attn)
        return attn_se


class SpatialAttention(nn.Module):

    def __init__(self, channels, kernel_size=31):
        super(SpatialAttention, self).__init__()
        self.channels = channels
        self.pool = nn.AdaptiveAvgPool2d(2)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.softmax = nn.Softmax(dim=1)
        self.conv2 = nn.Sequential(
            ReparamLargeKernelConvBNGELU(4, 4, large_kernel_size=kernel_size, groups=4),
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
        attn = self.conv2(attn)
        return attn


class ChannelSpatialAttention(nn.Module):

    def __init__(self, channels, kernel_size=17):
        super(ChannelSpatialAttention, self).__init__()
        self.cam = ChannelAttention(channels, kernel_size=kernel_size)
        self.sam = SpatialAttention(channels, kernel_size=kernel_size)
        self.fusion1 = ConvBNGELU(channels, channels, kernel_size=3, groups=channels)
        self.fusion2 = ConvBNGELU(channels, channels, kernel_size=3, groups=channels)

    def forward(self, x):
        attn_cam = self.cam(x)
        out = self.fusion1(x + attn_cam)
        attn_sam = self.sam(out)
        out = self.fusion2(out + attn_sam)
        return out
    

class UnifyChannels(nn.Module):

    def __init__(self, in_channels, channels):
        super().__init__()
        self.conv = nn.ModuleList()
        for i in in_channels:
            conv = nn.Conv2d(i, channels, kernel_size=1)
            self.conv.append(conv)
        self.attn = nn.ModuleList()
        for i in in_channels:
            attn = ChannelSpatialAttention(channels)
            self.attn.append(attn)

    def forward(self, x):
        out = []
        for _x, conv, attn in zip(x, self.conv, self.attn):
            out.append(attn(conv(_x)))
        return out


@HEADS.register_module()
class SFCRHead(BaseDecodeHead):

    def __init__(self, **kwargs):
        super(SFCRHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        self.unify_channels = UnifyChannels(self.in_channels, self.channels)
        self.sfn = StairFusionNetwork(self.channels, depth=3)

    def _forward_feature(self, inputs):
        inputs = self._transform_inputs(inputs)

        out = self.unify_channels(inputs)

        out = self.sfn(out)

        return out
 
    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
