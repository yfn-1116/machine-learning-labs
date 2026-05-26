import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


FIGURE_DIR = os.path.join("figures", "figures")


def ensure_figure_dir():
    os.makedirs(FIGURE_DIR, exist_ok=True)


def plot_class_distribution(y, filename="class_distribution.png"):
    """
    绘制类别分布图
    """
    ensure_figure_dir()

    counts = pd.Series(y).value_counts().sort_index()
    label_map = {
        0: "高复购用户(0)",
        1: "低活跃用户(1)",
        2: "流失风险用户(2)"
    }
    labels = [label_map.get(i, str(i)) for i in counts.index]

    plt.figure()
    plt.bar(labels, counts.values)
    plt.title("Class Distribution")
    plt.ylabel("Count")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, filename), dpi=300)
    plt.close()


def plot_missing_values(data, filename="missing_values.png"):
    """
    绘制各字段缺失值数量图
    """
    ensure_figure_dir()

    missing_counts = data.isnull().sum()

    plt.figure()
    plt.bar(missing_counts.index, missing_counts.values)
    plt.title("Missing Values Count")
    plt.ylabel("Missing Count")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, filename), dpi=300)
    plt.close()


def plot_accuracy_compare(id3_train_acc, id3_test_acc, c45_train_acc, c45_test_acc,
                          filename="accuracy_compare.png"):
    """
    绘制训练/测试准确率对比图
    """
    ensure_figure_dir()

    labels = ["ID3 Train", "ID3 Test", "C4.5 Train", "C4.5 Test"]
    values = [id3_train_acc, id3_test_acc, c45_train_acc, c45_test_acc]

    plt.figure()
    plt.bar(labels, values)
    plt.title("Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, filename), dpi=300)
    plt.close()


def plot_overfitting_compare(id3_gap, c45_gap, filename="overfitting_compare.png"):
    """
    绘制过拟合程度对比图（训练准确率 - 测试准确率）
    """
    ensure_figure_dir()

    labels = ["ID3", "C4.5"]
    values = [id3_gap, c45_gap]

    plt.figure()
    plt.bar(labels, values)
    plt.title("Overfitting Comparison")
    plt.ylabel("Train Accuracy - Test Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, filename), dpi=300)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, filename, title):
    """
    绘制混淆矩阵
    """
    ensure_figure_dir()

    plt.figure()
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["0", "1", "2"]
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, filename), dpi=300)
    plt.close()


def export_tree_graph(tree, filename, class_names=None):
    """
    使用 graphviz 导出决策树结构图。
    filename 传入如 "id3_tree"
    最终会生成 figures/figures/id3_tree.png
    """
    ensure_figure_dir()

    try:
        from graphviz import Digraph
    except ImportError:
        print("未安装 graphviz 的 Python 包，跳过决策树图片导出。")
        return

    dot = Digraph(comment="Decision Tree")
    dot.attr(rankdir="TB")

    counter = {"value": 0}

    if class_names is None:
        class_names = {
            0: "高复购用户",
            1: "低活跃用户",
            2: "流失风险用户"
        }

    def get_node_id():
        counter["value"] += 1
        return f"node_{counter['value']}"

    def add_nodes(node, parent_id=None, edge_label=""):
        node_id = get_node_id()

        if isinstance(node, dict):
            feature = list(node.keys())[0]
            dot.node(node_id, feature, shape="box")
            if parent_id is not None:
                dot.edge(parent_id, node_id, label=str(edge_label))

            for value, subtree in node[feature].items():
                add_nodes(subtree, node_id, str(value))
        else:
            leaf_label = class_names.get(node, str(node))
            dot.node(node_id, leaf_label, shape="ellipse")
            if parent_id is not None:
                dot.edge(parent_id, node_id, label=str(edge_label))

    add_nodes(tree)

    save_path = os.path.join(FIGURE_DIR, filename)
    try:
        dot.render(save_path, format="png", cleanup=True)
        print(f"决策树图已保存：{save_path}.png")
    except Exception as e:
        print("graphviz 渲染失败，可能系统未安装 graphviz 可执行程序。")
        print("错误信息：", e)
