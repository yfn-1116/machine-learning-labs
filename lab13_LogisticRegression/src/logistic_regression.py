import numpy as np
import pandas as pd


class NativeLogisticRegression:
    """
    逻辑回归原生实现。
    不调用sklearn.linear_model.LogisticRegression。

    核心功能：
    - Sigmoid概率输出
    - 交叉熵损失
    - 小批量梯度下降
    - L1/L2正则化
    - 类别不平衡权重
    - 特征权重解释
    """

    def __init__(
        self,
        learning_rate=0.05,
        n_iterations=1000,
        batch_size=32,
        regularization="l2",
        lambda_=0.01,
        class_weight="balanced",
        random_state=42,
        verbose=True,
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.regularization = regularization
        self.lambda_ = lambda_
        self.class_weight = class_weight
        self.random_state = random_state
        self.verbose = verbose

        self.weights = None
        self.bias = 0.0
        self.loss_history = []
        self.feature_names = None

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def _sample_weight(self, y):
        if self.class_weight != "balanced":
            return np.ones_like(y, dtype=float)

        n = len(y)
        pos = max(np.sum(y == 1), 1)
        neg = max(np.sum(y == 0), 1)

        w_pos = n / (2 * pos)
        w_neg = n / (2 * neg)
        return np.where(y == 1, w_pos, w_neg).astype(float)

    def _compute_loss(self, X, y, sample_weight):
        proba = self.predict_proba(X)
        eps = 1e-12
        proba = np.clip(proba, eps, 1 - eps)

        ce = -(y * np.log(proba) + (1 - y) * np.log(1 - proba))
        loss = np.average(ce, weights=sample_weight)

        if self.regularization == "l2":
            loss += self.lambda_ * np.sum(self.weights ** 2) / 2
        elif self.regularization == "l1":
            loss += self.lambda_ * np.sum(np.abs(self.weights))

        return float(loss)

    def fit(self, X, y, feature_names=None):
        rng = np.random.default_rng(self.random_state)
        y = np.asarray(y).astype(float)
        m, n = X.shape

        self.weights = np.zeros(n, dtype=float)
        self.bias = 0.0
        self.loss_history = []
        self.feature_names = feature_names if feature_names else [f"feature_{i}" for i in range(n)]

        sample_weight = self._sample_weight(y)

        for iteration in range(1, self.n_iterations + 1):
            indices = rng.permutation(m)

            for start in range(0, m, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                w_batch = sample_weight[batch_idx]

                z = X_batch @ self.weights + self.bias
                proba = self._sigmoid(z)
                error = proba - y_batch

                weighted_error = w_batch * error
                normalizer = np.sum(w_batch)

                dw = (X_batch.T @ weighted_error) / normalizer
                db = np.sum(weighted_error) / normalizer

                if self.regularization == "l2":
                    dw += self.lambda_ * self.weights
                elif self.regularization == "l1":
                    dw += self.lambda_ * np.sign(self.weights)

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            if iteration % 10 == 0:
                loss = self._compute_loss(X, y, sample_weight)
                self.loss_history.append(loss)

            if self.verbose and iteration % 100 == 0:
                print(f"迭代 {iteration}/{self.n_iterations}, 损失={self.loss_history[-1]:.6f}")

        return self

    def predict_proba(self, X):
        z = X @ self.weights + self.bias
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_feature_weights(self):
        """
        权重解释：
        - 权重 > 0：该特征越大，用户流失概率越高
        - 权重 < 0：该特征越大，用户流失概率越低
        - 绝对值越大：影响越强
        """
        return pd.DataFrame({
            "特征": self.feature_names,
            "权重": self.weights,
            "影响方向": np.where(self.weights >= 0, "增加流失风险", "降低流失风险"),
            "OddsRatio": np.exp(np.clip(self.weights, -20, 20)),
            "权重绝对值": np.abs(self.weights),
        }).sort_values("权重绝对值", ascending=False)
