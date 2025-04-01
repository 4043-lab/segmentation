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
将本文的代码加入到/mmseg/models/decode_heads/中，同时在configs文件夹创建相关运行配置。

- 项目运行，见mmsegmentation项目介绍。
- mmsegmentation 0.X版本可用，更新版本需要相应修改