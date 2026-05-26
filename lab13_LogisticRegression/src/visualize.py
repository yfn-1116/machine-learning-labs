import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def setup_chinese_font():
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC", "Noto Sans CJK JP", "SimHei", "Microsoft YaHei",
        "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _savefig(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_churn_distribution(y, save_path):
    setup_chinese_font()
    counts = [int(np.sum(y == 0)), int(np.sum(y == 1))]

    plt.figure(figsize=(6, 4))
    plt.bar(["留存", "流失"], counts)
    plt.title("用户流失分布")
    plt.xlabel("用户状态")
    plt.ylabel("用户数量")

    for i, v in enumerate(counts):
        plt.text(i, v, str(v), ha="center", va="bottom")

    _savefig(save_path)


def plot_loss_curve(loss_history, save_path):
    setup_chinese_font()

    plt.figure(figsize=(8, 5))
    x = np.arange(10, 10 * len(loss_history) + 1, 10)
    plt.plot(x, loss_history, marker="o")
    plt.title("原生逻辑回归训练损失下降曲线")
    plt.xlabel("迭代次数")
    plt.ylabel("交叉熵损失")
    plt.grid(alpha=0.3)

    _savefig(save_path)


def plot_confusion_matrix(cm, save_path):
    setup_chinese_font()

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("原生逻辑回归混淆矩阵")
    plt.colorbar()

    labels = ["留存", "流失"]
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")

    threshold = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    _savefig(save_path)


def plot_roc_curve(y_test, y_proba, save_path):
    setup_chinese_font()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC曲线")
    plt.xlabel("假阳性率 FPR")
    plt.ylabel("真正率 TPR")
    plt.legend()
    plt.grid(alpha=0.3)

    _savefig(save_path)


def plot_feature_weights(weights_df, save_path, top_n=12):
    setup_chinese_font()

    data = weights_df.head(top_n).sort_values("权重")
    plt.figure(figsize=(10, 7))
    plt.barh(data["特征"], data["权重"])
    plt.axvline(x=0, linestyle="--", linewidth=1)
    plt.title(f"Top {top_n} 特征权重")
    plt.xlabel("权重值：正数=增加流失风险，负数=降低流失风险")

    _savefig(save_path)


def plot_threshold_compare(threshold_df, save_path):
    setup_chinese_font()

    plt.figure(figsize=(8, 5))
    plt.plot(threshold_df["阈值"], threshold_df["精确率"], marker="o", label="精确率")
    plt.plot(threshold_df["阈值"], threshold_df["召回率"], marker="s", label="召回率")
    plt.plot(threshold_df["阈值"], threshold_df["F1"], marker="^", label="F1")
    plt.title("不同分类阈值下的精确率、召回率、F1对比")
    plt.xlabel("分类阈值")
    plt.ylabel("指标值")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()

    _savefig(save_path)


def plot_model_compare(compare_df, save_path):
    setup_chinese_font()

    metrics = ["准确率", "精确率", "召回率", "F1", "AUC"]
    x = np.arange(len(compare_df))
    width = 0.15

    plt.figure(figsize=(10, 5))
    for i, metric in enumerate(metrics):
        plt.bar(x + (i - 2) * width, compare_df[metric], width, label=metric)

    plt.xticks(x, compare_df["模型"])
    plt.ylim(0, 1.05)
    plt.title("原生逻辑回归与sklearn逻辑回归对比")
    plt.ylabel("指标值")
    plt.legend()

    _savefig(save_path)


def plot_risk_distribution(risk_df, save_path):
    setup_chinese_font()

    order = ["极低风险", "低风险", "中风险", "高风险", "极高风险"]
    counts = risk_df["风险等级"].value_counts().reindex(order).fillna(0)

    plt.figure(figsize=(8, 5))
    plt.bar(counts.index, counts.values)
    plt.title("用户流失风险等级分布")
    plt.xlabel("风险等级")
    plt.ylabel("用户数量")

    for i, v in enumerate(counts.values):
        plt.text(i, v, str(int(v)), ha="center", va="bottom")

    _savefig(save_path)
