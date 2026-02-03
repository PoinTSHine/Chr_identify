# 字符识别CNN模型项目

基于EMNIST数据集的字母识别卷积神经网络模型。

## 项目概述

该项目使用PyTorch构建了一个卷积神经网络(CNN)，用于识别EMNIST数据集中的字母。模型能够识别大写和小写字母，共52类。

## 功能特性

- 使用PyTorch构建的CNN模型
- 包含批量归一化和Dropout层以提高性能
- 数据增强技术提升模型泛化能力
- 支持GPU加速训练
- 模型可视化功能
- 混淆矩阵分析

## 模型架构

- 输入层：1x28x28 (灰度图像)
- 卷积层1：Conv2d(1, 32, 3x3) + BatchNorm + ReLU + MaxPool
- 卷积层2：Conv2d(32, 64, 3x3) + BatchNorm + ReLU + MaxPool
- 卷积层3：Conv2d(64, 128, 3x3) + BatchNorm + ReLU + MaxPool
- 全连接层1：Linear(1152, 256) + ReLU
- 输出层：Linear(256, 52)

## 依赖库

- torch
- torchvision
- torchviz
- matplotlib
- seaborn
- scikit-learn

## 使用方法

- 创建环境，安装所需依赖包

```bash
conda create -n letter_cnn python=3.10.19
conda activate letter_cnn
pip install -r requirements.txt
```

- 下载

```bash
git clone https://github.com/PoinTSHine/Chr_identify.git
cd Chr_identify
```

- 运行main.ipynb中的代码

- 训练完成

正确率最高的模型权重会保存为`best_model.pth`

模型结构会保存为`letter_cnn_torchviz.png`

混淆矩阵会保存为`confusion_matrix.png`

## 生成的文件

- `best_model.pth` - 训练好的最佳模型权重
- `confusion_matrix.png` - 混淆矩阵可视化
- `letter_cnn_torchviz.png` - 模型结构可视化

## 训练结果

模型在验证集上的准确率约为94.7%。
