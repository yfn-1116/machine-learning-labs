from collections import Counter
from pathlib import Path
import shutil
import time

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")

import matplotlib.pyplot as plt

try:
    import chineseize_matplotlib  # noqa: F401
except ModuleNotFoundError:
    chineseize_matplotlib = None

from datasets import (
    FEATURES,
    LABEL,
    load_or_create_data,
    preprocess_for_id3,
    preprocess_for_knn,
)


BASE_DIR = Path(__file__).resolve().parents[1]
FIG_DIR = BASE_DIR / "figures"
EXPERIMENT_RESULTS_PATH = FIG_DIR / "experiment_results.csv"
COLD_START_RESULTS_PATH = FIG_DIR / "cold_start_results.csv"
ID3_RULES_PATH = FIG_DIR / "id3_tree_rules.txt"
LEGACY_FIG_DIR = BASE_DIR / "src" / "figures"
LEGACY_EXPERIMENT_RESULTS_PATH = BASE_DIR / "src" / "experiment_results.csv"
LEGACY_COLD_START_RESULTS_PATH = BASE_DIR / "src" / "cold_start_results.csv"
OLD_ROOT_EXPERIMENT_RESULTS_PATH = BASE_DIR / "experiment_results.csv"
OLD_ROOT_COLD_START_RESULTS_PATH = BASE_DIR / "cold_start_results.csv"

FIG_DIR.mkdir(parents=True, exist_ok=True)


def migrate_legacy_outputs():
    """
    把历史上错误写入 src 目录的输出迁移到实验根目录。
    """
    legacy_to_current = [
        (LEGACY_EXPERIMENT_RESULTS_PATH, EXPERIMENT_RESULTS_PATH),
        (LEGACY_COLD_START_RESULTS_PATH, COLD_START_RESULTS_PATH),
        (OLD_ROOT_EXPERIMENT_RESULTS_PATH, EXPERIMENT_RESULTS_PATH),
        (OLD_ROOT_COLD_START_RESULTS_PATH, COLD_START_RESULTS_PATH),
    ]

    for legacy_path, current_path in legacy_to_current:
        if not current_path.exists() and legacy_path.exists():
            shutil.move(str(legacy_path), str(current_path))
        elif current_path.exists() and legacy_path.exists():
            legacy_path.unlink()

    if LEGACY_FIG_DIR.exists():
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        for legacy_file in LEGACY_FIG_DIR.iterdir():
            target_path = FIG_DIR / legacy_file.name
            if not target_path.exists():
                shutil.move(str(legacy_file), str(target_path))
            elif legacy_file.is_file():
                legacy_file.unlink()

        try:
            LEGACY_FIG_DIR.rmdir()
        except OSError:
            pass


class NativeKNN:
    """
    原生 KNN 算法。

    不调用 sklearn 的 KNeighborsClassifier，手动实现：
    1. 欧氏距离
    2. K 个最近邻选择
    3. 多数投票分类
    """

    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """
        KNN 没有真正的训练过程，只保存训练数据供预测时计算距离。
        """
        self.X_train = np.asarray(X_train, dtype=float)
        self.y_train = np.asarray(y_train)

    def _distance(self, x1, x2):
        """
        计算两个样本之间的欧氏距离。
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict_one(self, sample):
        """
        预测单个样本的类别。
        """
        distances = []
        for index in range(len(self.X_train)):
            dist = self._distance(sample, self.X_train[index])
            distances.append((dist, self.y_train[index]))

        distances.sort(key=lambda item: item[0])
        neighbors = distances[: self.k]
        labels = [label for _, label in neighbors]
        return Counter(labels).most_common(1)[0][0]

    def predict(self, X_test):
        """
        对测试集进行批量预测。
        """
        X_test = np.asarray(X_test, dtype=float)
        return np.array([self.predict_one(sample) for sample in X_test])


def standardize_train_test(X_train, X_test):
    """
    使用训练集统计量对训练集和测试集做标准化。
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std == 0, 1e-8, std)

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    return X_train_scaled, X_test_scaled


def entropy(y):
    """
    计算标签序列的信息熵。
    """
    counts = Counter(y)
    total = len(y)
    result = 0.0

    for count in counts.values():
        probability = count / total
        if probability > 0:
            result -= probability * np.log2(probability)

    return result


def information_gain(X, y, feature):
    """
    计算某个特征的信息增益。
    """
    total_entropy = entropy(y)
    weighted_entropy = 0.0

    for value in X[feature].unique():
        subset_y = y[X[feature] == value]
        weighted_entropy += len(subset_y) / len(y) * entropy(subset_y)

    return total_entropy - weighted_entropy


def majority_label(y):
    """
    返回样本中出现次数最多的类别。
    """
    return Counter(y).most_common(1)[0][0]


def build_id3_tree(X, y, features, min_samples_split=10):
    """
    递归构建 ID3 决策树。
    """
    if len(set(y)) == 1:
        return y.iloc[0]

    if not features or len(y) < min_samples_split:
        return majority_label(y)

    gains = [information_gain(X, y, feature) for feature in features]
    best_feature = features[int(np.argmax(gains))]
    remaining_features = [feature for feature in features if feature != best_feature]
    tree = {best_feature: {}}

    for value in X[best_feature].unique():
        sub_X = X[X[best_feature] == value]
        sub_y = y[X[best_feature] == value]

        if len(sub_y) == 0:
            tree[best_feature][value] = majority_label(y)
            continue

        tree[best_feature][value] = build_id3_tree(
            sub_X,
            sub_y,
            remaining_features,
            min_samples_split=min_samples_split,
        )

    return tree


def predict_id3_one(tree, sample, default_label=0):
    """
    使用 ID3 决策树预测单个样本。
    """
    if not isinstance(tree, dict):
        return tree

    feature = next(iter(tree))
    value = sample[feature]
    if value not in tree[feature]:
        return default_label

    return predict_id3_one(tree[feature][value], sample, default_label)


def predict_id3(tree, X_test, default_label=0):
    """
    对测试集做批量预测。
    """
    return np.array(
        [
            predict_id3_one(tree, X_test.iloc[index], default_label)
            for index in range(len(X_test))
        ]
    )


def evaluate(y_true, y_pred):
    """
    计算模型评价指标。
    """
    return {
        "准确率": accuracy_score(y_true, y_pred),
        "精确率": precision_score(y_true, y_pred, zero_division=0),
        "召回率": recall_score(y_true, y_pred, zero_division=0),
        "F1值": f1_score(y_true, y_pred, zero_division=0),
    }


def save_confusion_matrix(y_true, y_pred, title, filename):
    """
    保存混淆矩阵图片。
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.xticks([0, 1], ["非高意向", "高意向"])
    plt.yticks([0, 1], ["非高意向", "高意向"])

    for row_index in range(cm.shape[0]):
        for col_index in range(cm.shape[1]):
            plt.text(
                col_index, row_index, cm[row_index, col_index], ha="center", va="center"
            )

    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=150)
    plt.close()


def save_raw_scatter(data):
    """
    保存原始数据分布图。
    """
    plt.figure(figsize=(6, 4))
    plt.scatter(data["停留时长"], data["小黄车点击次数"], c=data["标签"], alpha=0.7)
    plt.title("直播用户行为原始分布")
    plt.xlabel("停留时长")
    plt.ylabel("小黄车点击次数")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "raw_vs_label.png", dpi=150)
    plt.close()


def run_basic_experiment():
    """
    运行基础实验：KNN 与 ID3 的训练、评估和结果保存。
    """
    data = preprocess_for_knn(load_or_create_data())
    save_raw_scatter(data)

    X = data[FEATURES]
    y = data[LABEL]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    knn_start_train = time.time()
    X_train_scaled, X_test_scaled = standardize_train_test(
        X_train.values, X_test.values
    )
    knn = NativeKNN(k=5)
    knn.fit(X_train_scaled, y_train.values)
    knn_train_time = time.time() - knn_start_train

    knn_start_pred = time.time()
    knn_pred = knn.predict(X_test_scaled)
    knn_pred_time = time.time() - knn_start_pred

    knn_result = evaluate(y_test, knn_pred)
    knn_result["训练时间"] = knn_train_time
    knn_result["预测时间"] = knn_pred_time
    save_confusion_matrix(y_test, knn_pred, "KNN 混淆矩阵", "knn_confusion_matrix.png")

    id3_data = preprocess_for_id3(data)
    X_id3 = id3_data.drop(columns=[LABEL])
    y_id3 = id3_data[LABEL]
    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
        X_id3,
        y_id3,
        test_size=0.3,
        random_state=42,
        stratify=y_id3,
    )

    id3_start_train = time.time()
    id3_tree = build_id3_tree(X_train_i, y_train_i, list(X_train_i.columns))
    id3_train_time = time.time() - id3_start_train

    id3_start_pred = time.time()
    id3_pred = predict_id3(id3_tree, X_test_i, default_label=majority_label(y_train_i))
    id3_pred_time = time.time() - id3_start_pred

    id3_result = evaluate(y_test_i, id3_pred)
    id3_result["训练时间"] = id3_train_time
    id3_result["预测时间"] = id3_pred_time
    save_confusion_matrix(
        y_test_i, id3_pred, "ID3 混淆矩阵", "id3_confusion_matrix.png"
    )

    with ID3_RULES_PATH.open("w", encoding="utf-8") as file:
        file.write(str(id3_tree))

    results = pd.DataFrame(
        [
            {
                "算法": "KNN",
                "核心参数": "K=5，欧氏距离，Z-Score标准化",
                **knn_result,
            },
            {
                "算法": "ID3",
                "核心参数": "无剪枝，停留时长3分桶，点击/互动分桶",
                **id3_result,
            },
        ]
    )
    results.to_csv(EXPERIMENT_RESULTS_PATH, index=False, encoding="utf-8-sig")

    return data, results


def run_cold_start_experiment(data):
    """
    冷启动实验：比较不同训练集比例下 KNN 和 ID3 的准确率变化。
    """
    ratios = [0.1, 0.3, 0.5, 0.7]
    knn_scores = []
    id3_scores = []

    X = data[FEATURES]
    y = data[LABEL]
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    id3_data = preprocess_for_id3(data)
    X_i = id3_data.drop(columns=[LABEL])
    y_i = id3_data[LABEL]
    X_train_i_full, X_test_i, y_train_i_full, y_test_i = train_test_split(
        X_i,
        y_i,
        test_size=0.3,
        random_state=42,
        stratify=y_i,
    )

    for ratio in ratios:
        sample_size = max(5, int(len(X_train_full) * ratio))

        X_train = X_train_full.iloc[:sample_size]
        y_train = y_train_full.iloc[:sample_size]
        X_train_scaled, X_test_scaled = standardize_train_test(
            X_train.values, X_test.values
        )

        knn = NativeKNN(k=5)
        knn.fit(X_train_scaled, y_train.values)
        knn_pred = knn.predict(X_test_scaled)
        knn_scores.append(accuracy_score(y_test, knn_pred))

        X_train_i = X_train_i_full.iloc[:sample_size]
        y_train_i = y_train_i_full.iloc[:sample_size]
        tree = build_id3_tree(
            X_train_i,
            y_train_i,
            list(X_train_i.columns),
            min_samples_split=5,
        )
        id3_pred = predict_id3(tree, X_test_i, default_label=majority_label(y_train_i))
        id3_scores.append(accuracy_score(y_test_i, id3_pred))

    cold_df = pd.DataFrame(
        {
            "训练集比例": ratios,
            "KNN准确率": knn_scores,
            "ID3准确率": id3_scores,
        }
    )
    cold_df.to_csv(COLD_START_RESULTS_PATH, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(6, 4))
    plt.plot(ratios, knn_scores, marker="o", label="KNN")
    plt.plot(ratios, id3_scores, marker="o", label="ID3")
    plt.title("冷启动场景准确率对比")
    plt.xlabel("训练集样本比例")
    plt.ylabel("准确率")
    plt.xticks(ratios, ["10%", "30%", "50%", "70%"])
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cold_start_accuracy.png", dpi=150)
    plt.close()


def main():
    """
    依次运行基础对比实验和冷启动实验。
    """
    migrate_legacy_outputs()
    data, results = run_basic_experiment()
    run_cold_start_experiment(data)

    print("实验完成。")
    print(f"结果文件：{EXPERIMENT_RESULTS_PATH.name}、{COLD_START_RESULTS_PATH.name}")
    print(f"图片目录：{FIG_DIR}")
    print()
    print(results)


if __name__ == "__main__":
    main()
