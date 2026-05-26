import numpy as np
from sklearn.metrics import accuracy_score


def _sse(y):
    if len(y) == 0:
        return 0.0
    return float(np.sum((y - np.mean(y)) ** 2))


class CARTRegressionTree:
    """
    原生CART回归树，作为GBDT的基学习器。
    不调用sklearn的DecisionTreeRegressor，方便报告里解释原生实现过程。
    """

    def __init__(self, max_depth=3, min_samples_split=6, min_samples_leaf=3, max_thresholds=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_thresholds = max_thresholds
        self.root_ = None
        self.feature_importances_ = None

    def fit(self, X, y):
        self.feature_importances_ = np.zeros(X.shape[1], dtype=float)
        self.root_ = self._build_tree(X, y, depth=0)
        return self

    def _candidate_thresholds(self, values):
        values = np.unique(values[~np.isnan(values)])
        if len(values) <= 1:
            return np.array([])
        if len(values) > self.max_thresholds:
            qs = np.linspace(0.05, 0.95, self.max_thresholds)
            return np.unique(np.quantile(values, qs))
        return values[:-1]

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        parent_sse = _sse(y)
        best_gain = 0.0
        best_feature = None
        best_threshold = None

        for fid in range(n_features):
            thresholds = self._candidate_thresholds(X[:, fid])
            for threshold in thresholds:
                left_mask = X[:, fid] <= threshold
                n_left = int(np.sum(left_mask))
                n_right = n_samples - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                gain = parent_sse - _sse(y[left_mask]) - _sse(y[~left_mask])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = fid
                    best_threshold = float(threshold)

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, depth):
        value = float(np.mean(y)) if len(y) else 0.0

        if depth >= self.max_depth or len(y) < self.min_samples_split or np.var(y) < 1e-12:
            return {"value": value}

        fid, threshold, gain = self._best_split(X, y)
        if fid is None or gain <= 1e-12:
            return {"value": value}

        left_mask = X[:, fid] <= threshold
        self.feature_importances_[fid] += gain

        return {
            "feature": fid,
            "threshold": threshold,
            "gain": gain,
            "value": value,
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._build_tree(X[~left_mask], y[~left_mask], depth + 1),
        }

    def _predict_one(self, node, x):
        while "feature" in node:
            if x[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["value"]

    def predict(self, X):
        return np.array([self._predict_one(self.root_, row) for row in X], dtype=float)


class BinaryGBDTClassifier:
    """
    二分类GBDT。
    核心逻辑：
    1. 初始化预测值
    2. 计算负梯度，也就是 y - p
    3. 用CART回归树拟合负梯度
    4. 按学习率更新累积预测
    """

    def __init__(self, n_estimators=20, learning_rate=0.2, max_depth=3, min_samples_split=6):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.base_score_ = 0.0
        self.estimators_ = []
        self.loss_history_ = []
        self.acc_history_ = []
        self.feature_importances_ = None

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    @staticmethod
    def _log_loss(y, p):
        eps = 1e-12
        p = np.clip(p, eps, 1.0 - eps)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def fit(self, X, y):
        y = y.astype(float)
        pos_rate = float(np.clip(np.mean(y), 1e-6, 1 - 1e-6))
        self.base_score_ = np.log(pos_rate / (1 - pos_rate))
        F = np.full(len(y), self.base_score_, dtype=float)

        self.estimators_ = []
        self.loss_history_ = []
        self.acc_history_ = []
        self.feature_importances_ = np.zeros(X.shape[1], dtype=float)

        for _ in range(self.n_estimators):
            p = self._sigmoid(F)
            negative_gradient = y - p

            tree = CARTRegressionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=max(3, self.min_samples_split // 2),
            ).fit(X, negative_gradient)

            update = tree.predict(X)
            F += self.learning_rate * update

            self.estimators_.append(tree)
            self.feature_importances_ += tree.feature_importances_

            p_new = self._sigmoid(F)
            self.loss_history_.append(self._log_loss(y, p_new))
            self.acc_history_.append(accuracy_score(y.astype(int), (p_new >= 0.5).astype(int)))

        total_gain = self.feature_importances_.sum()
        if total_gain > 0:
            self.feature_importances_ = self.feature_importances_ / total_gain

        return self

    def predict_proba_positive(self, X):
        F = np.full(X.shape[0], self.base_score_, dtype=float)
        for tree in self.estimators_:
            F += self.learning_rate * tree.predict(X)
        return self._sigmoid(F)


class GBDTMultiClass:
    """
    多分类GBDT。
    使用 One-vs-Rest 方式支持：
    低转化 / 普通转化 / 高转化 / 爆款转化。
    """

    def __init__(self, n_estimators=20, learning_rate=0.2, max_depth=3, min_samples_split=6):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.classes_ = None
        self.classifiers_ = []
        self.feature_importances_ = None
        self.loss_history_ = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.classifiers_ = []
        histories = []
        importances = []

        for cls in self.classes_:
            y_binary = (y == cls).astype(int)
            clf = BinaryGBDTClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            ).fit(X, y_binary)

            self.classifiers_.append(clf)
            histories.append(clf.loss_history_)
            importances.append(clf.feature_importances_)

        self.loss_history_ = np.mean(np.array(histories), axis=0).tolist()
        self.feature_importances_ = np.mean(np.array(importances), axis=0)

        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ = self.feature_importances_ / total

        return self

    def predict_proba(self, X):
        probas = np.vstack([clf.predict_proba_positive(X) for clf in self.classifiers_]).T
        row_sum = probas.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return probas / row_sum

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
