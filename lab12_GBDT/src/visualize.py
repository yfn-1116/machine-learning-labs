from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score


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


def plot_class_distribution(y, class_names, save_path):
    setup_chinese_font()
    counts = [int(np.sum(y == i)) for i in range(len(class_names))]
    plt.figure(figsize=(8, 5))
    plt.bar(class_names, counts)
    plt.title("转化率等级样本分布")
    plt.xlabel("转化率等级")
    plt.ylabel("样本数量")
    for i, v in enumerate(counts):
        plt.text(i, v, str(v), ha="center", va="bottom")
    _savefig(save_path)


def plot_confusion_matrix(cm, class_names, title, save_path):
    setup_chinese_font()
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=30, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    threshold = cm.max() / 2 if cm.max() else 0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    _savefig(save_path)


def plot_feature_importance(importances, feature_names, save_path):
    setup_chinese_font()
    order = np.argsort(importances)[::-1]
    plt.figure(figsize=(11, 6))
    plt.bar(range(len(order)), importances[order])
    plt.xticks(range(len(order)), [feature_names[i] for i in order], rotation=45, ha="right")
    plt.title("原生GBDT特征重要性")
    plt.xlabel("直播运营指标")
    plt.ylabel("重要性权重")
    _savefig(save_path)


def plot_model_compare(results_df, save_path):
    setup_chinese_font()
    metrics = ["训练准确率", "测试准确率", "爆款召回率"]
    x = np.arange(len(results_df))
    width = 0.24

    plt.figure(figsize=(9, 5))
    for idx, metric in enumerate(metrics):
        plt.bar(x + (idx - 1) * width, results_df[metric], width, label=metric)

    plt.xticks(x, results_df["模型"])
    plt.ylim(0, 1.05)
    plt.title("随机森林与原生GBDT核心指标对比")
    plt.ylabel("指标值")
    plt.legend()
    _savefig(save_path)


def plot_overfitting_compare(results_df, save_path):
    setup_chinese_font()
    gap = results_df["训练准确率"] - results_df["测试准确率"]
    plt.figure(figsize=(8, 5))
    plt.bar(results_df["模型"], gap)
    plt.title("过拟合程度对比：训练准确率 - 测试准确率")
    plt.xlabel("模型")
    plt.ylabel("准确率差值")

    for i, v in enumerate(gap):
        plt.text(i, v, f"{v:.4f}", ha="center", va="bottom")

    _savefig(save_path)


def plot_pr_curve_compare(y_test, proba_dict, class_index, class_name, save_path):
    setup_chinese_font()
    y_binary = (y_test == class_index).astype(int)

    plt.figure(figsize=(8, 6))
    for model_name, proba in proba_dict.items():
        scores = proba[:, class_index]
        precision, recall, _ = precision_recall_curve(y_binary, scores)
        ap = average_precision_score(y_binary, scores)
        plt.plot(recall, precision, label=f"{model_name} AP={ap:.4f}")

    plt.title(f"{class_name}类别 PR 曲线对比")
    plt.xlabel("Recall / 召回率")
    plt.ylabel("Precision / 精确率")
    plt.legend()
    plt.grid(alpha=0.3)
    _savefig(save_path)


def plot_loss_curve(loss_history, save_path):
    setup_chinese_font()
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker="o")
    plt.title("原生GBDT训练损失变化")
    plt.xlabel("迭代轮数 / 树数量")
    plt.ylabel("平均对数损失")
    plt.grid(alpha=0.3)
    _savefig(save_path)


def plot_hyperparam_compare(hyper_df, save_path):
    setup_chinese_font()
    labels = hyper_df["参数组合"]

    plt.figure(figsize=(10, 5))
    plt.plot(labels, hyper_df["测试准确率"], marker="o", label="测试准确率")
    plt.plot(labels, hyper_df["爆款召回率"], marker="s", label="爆款召回率")
    plt.xticks(rotation=35, ha="right")
    plt.ylim(0, 1.05)
    plt.title("GBDT超参数对性能的影响")
    plt.xlabel("参数组合")
    plt.ylabel("指标值")
    plt.legend()
    plt.grid(alpha=0.3)
    _savefig(save_path)
