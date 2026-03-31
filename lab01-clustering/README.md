# 机器学习实验 1：Scikit-Learn的学习与运用（聚类算法实践）

## 实验概述
完成一个基于Scikit-Learn的无监督学习中聚类算法实践的实验，核心目标是通过Scikit-Learn工具掌握K-均值（K-Means）聚类的原理、实现、评估方法，同时对比层次聚类、DBSCAN等其他聚类算法的差异，掌握机器学习原生环境的搭建、配置与使用。

## 项目结构
本项目为系列机器学习实验的第一个实验（聚类算法实践），每个实验位于独立的子目录中，便于管理和扩展。

- `lab01-clustering/`：聚类算法实验
  - `lab01_clustering.py`：实验主脚本
  - `figures/`：生成的图表文件

后续实验将按照相同结构组织。

## 实验目标
- 掌握Python机器学习开发环境的搭建（Scikit-Learn、NumPy、Matplotlib等基础包）
- 理解K-Means算法的迭代原理（初始化质心→样本分配→质心更新→收敛）
- 学会使用Scikit-Learn实现K-Means聚类，并用轮廓系数、簇内平方和评估聚类效果
- 掌握"肘部法则""轮廓系数曲线"确定最佳簇数K的方法
- 对比K-Means、层次聚类、DBSCAN的聚类结果，理解不同算法的适用场景与局限性

## 环境要求
- Python 3.8及以上
- 所需包：scikit-learn, numpy, pandas, matplotlib, seaborn, scipy

## 快速开始

### 安装依赖
```bash
pip install scikit-learn numpy pandas matplotlib seaborn scipy
```

### 运行实验
```bash
python lab01-clustering/lab01_clustering.py
```

### 验证安装
```python
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(f"Scikit-Learn版本：{sklearn.__version__}")
print("所有包导入成功！")
```

## 实验步骤
1. **数据集准备**：生成模拟数据集（make_blobs）和加载Iris数据集
2. **K-Means模型构建与训练**：对两个数据集分别训练K-Means模型
3. **结果分析与评估**：可视化聚类结果，计算轮廓系数和调整兰德指数
4. **最佳簇数K的确定**：通过肘部法则和轮廓系数曲线确定最佳K
5. **扩展实践**：对比层次聚类和DBSCAN与K-Means的差异

## 实验内容
- 模拟数据集：生成3个高斯分布的簇，n_samples=500，n_features=2
- Iris数据集：真实数据集，4维特征，3个类别
- 数据标准化：使用StandardScaler消除特征尺度影响
- PCA降维：将Iris数据集从4维降至2维便于可视化
- 聚类算法：K-Means、层次聚类（AgglomerativeClustering）、DBSCAN
- 评估指标：轮廓系数、簇内平方和（Inertia）、调整兰德指数（ARI）

## 预期结果
- 生成7张可视化图表，展示数据分布、聚类结果和评估曲线
- 输出聚类评估指标和算法对比结果
- 通过控制台输出详细的分析结果

## 文件说明
- `lab01-clustering/lab01_clustering.py`：主实验脚本，包含所有实验步骤
- `requirements.txt`：依赖包列表
- `README.md`：本说明文件

## 注意事项
- 实验使用固定随机种子（random_state=42）确保结果可复现
- K-Means算法对数据尺度敏感，必须进行标准化
- DBSCAN算法对eps和min_samples参数敏感，需要适当调整
- 实验代码包含详细注释，便于理解和学习