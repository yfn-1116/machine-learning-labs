# 机器学习实验 8：KNN 与 ID3 算法

## 一、项目说明

本项目用于完成机器学习实验 8：KNN 与 ID3 算法。

实验场景是直播电商用户意向预测，根据用户的行为数据判断用户是否为高意向下单用户。

使用的特征包括：

- 停留时长
- 小黄车点击次数
- 互动次数

预测标签：

- 1：高意向用户
- 0：非高意向用户

---

## 二、项目结构

```text
lab08_KNN_ID3/
├── data/
│   └── live_ecommerce_user_intention.csv
├── figures/
│   ├── raw_vs_label.png
│   ├── knn_confusion_matrix.png
│   ├── id3_confusion_matrix.png
│   ├── cold_start_accuracy.png
│   └── id3_tree_rules.txt
├── src/
│   ├── datasets.py
│   └── experiment.py
├── requirements.txt
└── README.md