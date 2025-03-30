# Content-guided and Class-oriented Learning forVHR Image Semantic Segmentation论文主干代码

## 代码环境介绍
- 实验所用数据集为ISPRS Vaihingen、ISPRS Potsdam 和LoveDA数据集。
LoveDA数据集：
```
https://zenodo.org/records/5706578#.Yi2m1-hByUk
```
- 实验所用环境为mmsegmentation，项目介绍
```
https://github.com/open-mmlab/mmsegmentation
```
将本文的代码加入到/mmseg/models/decode_heads/中，同时在configs文件夹中新建一个文件夹，用于防止定义实验参数，包括batch_size、图像增强方法等的config文件。
- 实验中用到的可变形卷积DCNv3，其项目代码地址为
```
https://github.com/OpenGVLab/InternImage
```
根据其项目进行安装后，可以使用DCNv3

- 项目运行，见mmsegmentation项目介绍。
