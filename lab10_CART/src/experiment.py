# -*- coding: utf-8 -*-
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Noto Sans CJK JP", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from datasets import (
    FEATURE_NAMES,
    TARGET_NAME,
    CLASS_NAMES,
    load_dataset,
    preprocess_dataset,
    split_train_test,
    show_basic_info,
)

from cart import (
    build_cart_tree,
    predict,
    print_rules,
    plot_tree,
    count_leaves,
    tree_depth,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "gym_user_behavior.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"


def ensure_dirs():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)


def plot_class_distribution(y, save_path):
    counts = y.value_counts().sort_index()
    labels = [CLASS_NAMES.get(int(i), str(i)) for i in counts.index]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, counts.values)
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")

    for i, v in enumerate(counts.values):
        plt.text(i, v, str(v), ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[已保存] 用户类别分布图：{save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path):
    labels = sorted(CLASS_NAMES.keys())
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)

    ax.set_title("CART Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    tick_labels = [CLASS_NAMES[i] for i in labels]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(tick_labels, rotation=30, ha="right")
    ax.set_yticklabels(tick_labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[已保存] 混淆矩阵：{save_path}")


def plot_metrics_table(y_true, y_pred, train_acc, test_acc, train_time, pred_time, save_path):
    labels = sorted(CLASS_NAMES.keys())
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    rows = []
    for idx, label in enumerate(labels):
        rows.append([
            CLASS_NAMES[label],
            f"{precision[idx]:.4f}",
            f"{recall[idx]:.4f}",
            f"{f1[idx]:.4f}",
            int(support[idx]),
        ])

    rows.append(["训练集准确率", f"{train_acc:.4f}", "", "", ""])
    rows.append(["测试集准确率", f"{test_acc:.4f}", "", "", ""])
    rows.append(["训练时间(s)", f"{train_time:.4f}", "", "", ""])
    rows.append(["预测时间(s)", f"{pred_time:.4f}", "", "", ""])

    columns = ["指标/类别", "Precision", "Recall", "F1-score", "Support"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    ax.set_title("CART Metrics Table", fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[已保存] 评估指标表：{save_path}")


def run_depth_comparison(X_train, y_train, X_test, y_test, save_path):
    depths = list(range(1, 8))
    train_acc_list = []
    test_acc_list = []
    rule_count_list = []

    for depth in depths:
        tree = build_cart_tree(
            X_train,
            y_train,
            max_depth=depth,
            min_samples_split=5,
            min_samples_leaf=2,
            min_gain=1e-6,
        )

        y_train_pred = predict(tree, X_train)
        y_test_pred = predict(tree, X_test)

        train_acc_list.append(accuracy_score(y_train, y_train_pred))
        test_acc_list.append(accuracy_score(y_test, y_test_pred))
        rule_count_list.append(count_leaves(tree))

    plt.figure(figsize=(8, 5))
    plt.plot(depths, train_acc_list, marker="o", label="Train Accuracy")
    plt.plot(depths, test_acc_list, marker="o", label="Test Accuracy")
    plt.title("Overfitting Comparison by max_depth")
    plt.xlabel("max_depth")
    plt.ylabel("Accuracy")
    plt.xticks(depths)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[已保存] 过拟合对比图：{save_path}")

    csv_path = save_path.with_suffix(".csv")
    pd.DataFrame({
        "max_depth": depths,
        "train_accuracy": train_acc_list,
        "test_accuracy": test_acc_list,
        "rule_count": rule_count_list,
    }).to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"[已保存] 过拟合对比数据：{csv_path}")


def main():
    ensure_dirs()

    print("========== 机器学习实验10：CART 算法原生实现 ==========")
    print(f"项目根目录：{PROJECT_ROOT}")
    print(f"图片输出目录：{FIGURES_DIR}")

    df = load_dataset(DATA_PATH)
    show_basic_info(df)

    plot_class_distribution(df[TARGET_NAME], FIGURES_DIR / "cart_class_distribution.png")

    X, y, encoders = preprocess_dataset(df)

    print("\n========== 类别特征编码映射 ==========")
    for col, mapping in encoders.items():
        print(f"{col}: {mapping}")

    X_train_df, X_test_df, y_train_ser, y_test_ser = split_train_test(
        X,
        y,
        test_size=0.3,
        random_state=42,
    )

    X_train = X_train_df.values.astype(float)
    X_test = X_test_df.values.astype(float)
    y_train = y_train_ser.values.astype(int)
    y_test = y_test_ser.values.astype(int)

    print("\n========== 训练集/测试集规模 ==========")
    print(f"训练集：{X_train.shape[0]} 条")
    print(f"测试集：{X_test.shape[0]} 条")

    print("\n========== 开始训练 CART 原生模型 ==========")
    start_train = time.perf_counter()

    cart_tree = build_cart_tree(
        X_train,
        y_train,
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=2,
        min_gain=1e-6,
    )

    train_time = time.perf_counter() - start_train

    start_pred = time.perf_counter()
    y_train_pred = predict(cart_tree, X_train)
    y_test_pred = predict(cart_tree, X_test)
    pred_time = time.perf_counter() - start_pred

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print("\n========== CART 模型评估 ==========")
    print(f"训练集准确率：{train_acc:.4f}")
    print(f"测试集准确率：{test_acc:.4f}")
    print(f"训练时间：{train_time:.4f} s")
    print(f"预测时间：{pred_time:.4f} s")
    print(f"树深度：{tree_depth(cart_tree)}")
    print(f"决策规则数量：{count_leaves(cart_tree)}")

    print("\n========== 分类报告 ==========")
    target_names = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())]
    print(classification_report(y_test, y_test_pred, target_names=target_names, zero_division=0))

    print("\n========== CART 用户划分规则 ==========")
    rules_text = print_rules(cart_tree, FEATURE_NAMES, CLASS_NAMES)

    rules_path = FIGURES_DIR / "cart_rules.txt"
    rules_path.write_text(rules_text, encoding="utf-8")
    print(f"[已保存] CART 决策规则文本：{rules_path}")

    plot_tree(cart_tree, FEATURE_NAMES, CLASS_NAMES, FIGURES_DIR / "cart_tree.png")
    plot_confusion_matrix(y_test, y_test_pred, FIGURES_DIR / "cart_confusion_matrix.png")
    plot_metrics_table(y_test, y_test_pred, train_acc, test_acc, train_time, pred_time, FIGURES_DIR / "cart_metrics_table.png")
    run_depth_comparison(X_train, y_train, X_test, y_test, FIGURES_DIR / "cart_overfitting_depth.png")

    print("\n========== 实验完成 ==========")
    print("已生成文件：")
    for p in sorted(FIGURES_DIR.iterdir()):
        print(p)


if __name__ == "__main__":
    main()
