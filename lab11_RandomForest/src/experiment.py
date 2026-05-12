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

from cart_base import (
    build_cart_tree,
    predict as cart_predict,
    print_rules,
    count_leaves,
    tree_depth,
)

from random_forest import (
    RandomForest,
    plot_feature_importance,
    plot_estimator_comparison,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "sku_behavior.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"


def ensure_dirs():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)


def plot_class_distribution(y, save_path):
    counts = y.value_counts().sort_index()
    labels = [CLASS_NAMES.get(int(i), str(i)) for i in counts.index]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, counts.values, color=plt.cm.Set2(np.linspace(0, 1, len(labels))))
    plt.title("SKU Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")

    for bar, v in zip(bars, counts.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2, v,
            str(v), ha="center", va="bottom",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[已保存] SKU类别分布图：{save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path, title_prefix=""):
    labels = sorted(CLASS_NAMES.keys())
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_title(f"{title_prefix}Confusion Matrix")
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


def plot_metrics_table(
    y_true, y_pred, train_acc, test_acc, train_time, pred_time, save_path, title_prefix=""
):
    labels = sorted(CLASS_NAMES.keys())
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0,
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
    ax.set_title(f"{title_prefix}Metrics Table", fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[已保存] 评估指标表：{save_path}")


def run_manual_rule_classification(X_df):
    margin = X_df["毛利率"].values
    sales = X_df["30天销量"].values
    y_pred = np.zeros(len(X_df), dtype=int)

    for i in range(len(X_df)):
        if sales[i] > 200:
            y_pred[i] = 0
        elif margin[i] > 0.30:
            y_pred[i] = 1
        elif margin[i] > 0.50 and sales[i] < 30:
            y_pred[i] = 2
        else:
            y_pred[i] = 3
    return y_pred


def main():
    ensure_dirs()

    print("========== 机器学习实验11：随机森林算法原生实现 ==========")
    print(f"项目根目录：{PROJECT_ROOT}")
    print(f"图片输出目录：{FIGURES_DIR}")

    df = load_dataset(DATA_PATH)
    show_basic_info(df)

    plot_class_distribution(df[TARGET_NAME], FIGURES_DIR / "rf_class_distribution.png")

    X, y = preprocess_dataset(df)

    X_train_df, X_test_df, y_train_ser, y_test_ser = split_train_test(
        X, y, test_size=0.3, random_state=42,
    )

    X_train = X_train_df.values.astype(float)
    X_test = X_test_df.values.astype(float)
    y_train = y_train_ser.values.astype(int)
    y_test = y_test_ser.values.astype(int)

    print("\n========== 训练集/测试集规模 ==========")
    print(f"训练集：{X_train.shape[0]} 条")
    print(f"测试集：{X_test.shape[0]} 条")

    # ==================== 1. 人工经验规则 ====================
    print("\n========== 1. 人工经验规则分类 ==========")
    start_manual = time.perf_counter()
    y_train_pred_manual = run_manual_rule_classification(X_train_df)
    y_test_pred_manual = run_manual_rule_classification(X_test_df)
    manual_time = time.perf_counter() - start_manual
    manual_train_acc = accuracy_score(y_train, y_train_pred_manual)
    manual_test_acc = accuracy_score(y_test, y_test_pred_manual)
    print(f"训练集准确率：{manual_train_acc:.4f}")
    print(f"测试集准确率：{manual_test_acc:.4f}")

    # ==================== 2. 单棵 CART 树 ====================
    print("\n========== 2. 单棵 CART 决策树 ==========")
    start_train = time.perf_counter()
    cart_tree = build_cart_tree(
        X_train, y_train,
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=2,
    )
    cart_train_time = time.perf_counter() - start_train

    start_pred = time.perf_counter()
    y_train_pred_cart = cart_predict(cart_tree, X_train)
    y_test_pred_cart = cart_predict(cart_tree, X_test)
    cart_pred_time = time.perf_counter() - start_pred

    cart_train_acc = accuracy_score(y_train, y_train_pred_cart)
    cart_test_acc = accuracy_score(y_test, y_test_pred_cart)

    print(f"训练集准确率：{cart_train_acc:.4f}")
    print(f"测试集准确率：{cart_test_acc:.4f}")
    print(f"训练时间：{cart_train_time:.4f} s")
    print(f"预测时间：{cart_pred_time:.4f} s")
    print(f"树深度：{tree_depth(cart_tree)}")
    print(f"决策规则数量：{count_leaves(cart_tree)}")

    print("\n========== CART 货品划分规则 ==========")
    cart_rules = print_rules(cart_tree, FEATURE_NAMES, CLASS_NAMES)
    (FIGURES_DIR / "cart_rules.txt").write_text(cart_rules, encoding="utf-8")
    print(f"[已保存] CART 决策规则文本")

    # ==================== 3. 随机森林 ====================
    print("\n========== 3. 随机森林 ==========")
    start_train = time.perf_counter()
    rf = RandomForest(
        n_estimators=15,
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features=None,
        random_state=42,
    )
    rf.fit(X_train, y_train, feature_names=FEATURE_NAMES)
    rf_train_time = time.perf_counter() - start_train

    start_pred = time.perf_counter()
    y_train_pred_rf = rf.predict(X_train)
    y_test_pred_rf = rf.predict(X_test)
    rf_pred_time = time.perf_counter() - start_pred

    rf_train_acc = accuracy_score(y_train, y_train_pred_rf)
    rf_test_acc = accuracy_score(y_test, y_test_pred_rf)

    print(f"训练集准确率：{rf_train_acc:.4f}")
    print(f"测试集准确率：{rf_test_acc:.4f}")
    print(f"训练时间：{rf_train_time:.4f} s")
    print(f"预测时间：{rf_pred_time:.4f} s")

    print("\n========== 特征重要性排序 ==========")
    indices = np.argsort(rf.feature_importances_)[::-1]
    for i in indices:
        print(f"  {FEATURE_NAMES[i]}: {rf.feature_importances_[i]:.4f}")

    # ==================== 4. 分类报告对比 ====================
    target_names = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())]

    print("\n========== 人工规则 分类报告 ==========")
    print(classification_report(y_test, y_test_pred_manual, target_names=target_names, zero_division=0))

    print("\n========== CART 分类报告 ==========")
    print(classification_report(y_test, y_test_pred_cart, target_names=target_names, zero_division=0))

    print("\n========== 随机森林 分类报告 ==========")
    print(classification_report(y_test, y_test_pred_rf, target_names=target_names, zero_division=0))

    # ==================== 5. 全店货品结构分布 ====================
    print("\n========== 全店货品结构分布（随机森林） ==========")
    all_pred = rf.predict(X.values.astype(float))
    from collections import Counter
    counts = Counter(all_pred)
    for cls_id, cnt in counts.items():
        print(f"  {CLASS_NAMES[cls_id]}: {cnt} 个SKU, 占比 {cnt / len(all_pred) * 100:.1f}%")

    # ==================== 6. 保存可视化 ====================
    print("\n========== 保存可视化结果 ==========")

    plot_confusion_matrix(y_test, y_test_pred_manual, FIGURES_DIR / "rf_confusion_manual.png", "Manual ")
    plot_confusion_matrix(y_test, y_test_pred_cart, FIGURES_DIR / "rf_confusion_cart.png", "CART ")
    plot_confusion_matrix(y_test, y_test_pred_rf, FIGURES_DIR / "rf_confusion_rf.png", "RF ")

    plot_metrics_table(
        y_test, y_test_pred_rf, rf_train_acc, rf_test_acc,
        rf_train_time, rf_pred_time,
        FIGURES_DIR / "rf_metrics_table.png", "RF ",
    )

    plot_feature_importance(rf.feature_importances_, FEATURE_NAMES, FIGURES_DIR / "rf_feature_importance.png")

    plot_estimator_comparison(
        X_train, y_train, X_test, y_test,
        estimator_range=[1, 3, 5, 7, 10, 15, 20, 25],
        save_path=FIGURES_DIR / "rf_estimator_comparison.png",
        rf_kwargs={"max_depth": 4},
    )

    # ==================== 7. 对比汇总表 ====================
    print("\n========== 三种方式对比汇总 ==========")
    print(f"{'方式':<16} {'训练准确率':<12} {'测试准确率':<12} {'时间(s)':<10}")
    print("-" * 50)
    print(f"{'人工规则':<16} {manual_train_acc:<12.4f} {manual_test_acc:<12.4f} {manual_time:<10.4f}")
    print(f"{'单CART树':<16} {cart_train_acc:<12.4f} {cart_test_acc:<12.4f} {cart_train_time + cart_pred_time:<10.4f}")
    print(f"{'随机森林':<16} {rf_train_acc:<12.4f} {rf_test_acc:<12.4f} {rf_train_time + rf_pred_time:<10.4f}")

    print("\n========== 实验完成 ==========")
    print("已生成文件：")
    for p in sorted(FIGURES_DIR.iterdir()):
        print(f"  {p}")


if __name__ == "__main__":
    main()
