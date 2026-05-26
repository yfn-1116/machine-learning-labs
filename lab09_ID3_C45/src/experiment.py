import os
import time
from sklearn.metrics import accuracy_score, classification_report

from datasets import (
    load_dataset,
    fill_missing_for_id3,
    discretize_for_id3,
    split_data
)
from id3_tree import ID3Tree
from c45_tree import C45Tree
from visualize import (
    plot_class_distribution,
    plot_missing_values,
    plot_accuracy_compare,
    plot_overfitting_compare,
    plot_confusion_matrix,
    export_tree_graph
)


DATA_PATH = "data/gym_user_data.csv"


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"找不到数据集文件：{DATA_PATH}")

    raw_data = load_dataset(DATA_PATH)

    # =========================
    # 1. 数据可视化（原始数据）
    # =========================
    plot_class_distribution(raw_data["标签"], filename="class_distribution.png")
    plot_missing_values(raw_data, filename="missing_values.png")

    # =========================
    # 2. ID3 数据处理
    # =========================
    id3_data = fill_missing_for_id3(raw_data)
    id3_data = discretize_for_id3(id3_data)

    X_train_id3, X_test_id3, y_train_id3, y_test_id3 = split_data(id3_data)

    # =========================
    # 3. C4.5 数据处理
    # =========================
    X_train_c45, X_test_c45, y_train_c45, y_test_c45 = split_data(raw_data)

    # =========================
    # 4. 训练 ID3
    # =========================
    id3 = ID3Tree()

    start = time.time()
    id3.fit(X_train_id3, y_train_id3)
    id3_train_time = time.time() - start

    start = time.time()
    id3_train_pred = id3.predict(X_train_id3)
    id3_test_pred = id3.predict(X_test_id3)
    id3_predict_time = time.time() - start

    id3_train_acc = accuracy_score(y_train_id3, id3_train_pred)
    id3_test_acc = accuracy_score(y_test_id3, id3_test_pred)

    # =========================
    # 5. 训练 C4.5
    # =========================
    c45 = C45Tree(min_samples=5, max_depth=5, min_gain_ratio=0.01)

    start = time.time()
    c45.fit(X_train_c45, y_train_c45)
    c45_train_time = time.time() - start

    start = time.time()
    c45_train_pred = c45.predict(X_train_c45)
    c45_test_pred = c45.predict(X_test_c45)
    c45_predict_time = time.time() - start

    c45_train_acc = accuracy_score(y_train_c45, c45_train_pred)
    c45_test_acc = accuracy_score(y_test_c45, c45_test_pred)

    # =========================
    # 6. 打印结果
    # =========================
    print("========== ID3 实验结果 ==========")
    print("训练集准确率:", round(id3_train_acc, 4))
    print("测试集准确率:", round(id3_test_acc, 4))
    print("训练时间:", round(id3_train_time, 4), "s")
    print("预测时间:", round(id3_predict_time, 4), "s")
    print(classification_report(y_test_id3, id3_test_pred, digits=4, zero_division=0))
    print()

    print("========== C4.5 实验结果 ==========")
    print("训练集准确率:", round(c45_train_acc, 4))
    print("测试集准确率:", round(c45_test_acc, 4))
    print("训练时间:", round(c45_train_time, 4), "s")
    print("预测时间:", round(c45_predict_time, 4), "s")
    print(classification_report(y_test_c45, c45_test_pred, digits=4, zero_division=0))
    print()

    print("========== ID3 决策树 ==========")
    print(id3.tree)
    print()

    print("========== C4.5 决策树 ==========")
    print(c45.tree)
    print()

    # =========================
    # 7. 结果可视化
    # =========================
    plot_accuracy_compare(
        id3_train_acc,
        id3_test_acc,
        c45_train_acc,
        c45_test_acc,
        filename="accuracy_compare.png"
    )

    plot_overfitting_compare(
        id3_train_acc - id3_test_acc,
        c45_train_acc - c45_test_acc,
        filename="overfitting_compare.png"
    )

    plot_confusion_matrix(
        y_test_id3,
        id3_test_pred,
        filename="id3_confusion_matrix.png",
        title="ID3 Confusion Matrix"
    )

    plot_confusion_matrix(
        y_test_c45,
        c45_test_pred,
        filename="c45_confusion_matrix.png",
        title="C4.5 Confusion Matrix"
    )

    # 决策树图（需要 graphviz 可用）
    export_tree_graph(id3.tree, filename="id3_tree")
    export_tree_graph(c45.tree, filename="c45_tree")

    print("所有可视化图片已输出到：figures/figures/")


if __name__ == "__main__":
    main()
