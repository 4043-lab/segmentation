# Stair fusion network with context-refined attention for remote sensing image semantic segmentation论文主干代码

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

- 项目运行，见mmsegmentation项目介绍。