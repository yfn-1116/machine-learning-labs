# -*- coding: utf-8 -*-
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Noto Sans CJK JP", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from cart_base import build_cart_tree, predict_one


class RandomForest:
    def __init__(
        self,
        n_estimators=15,
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features=None,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.feature_importances_ = None
        self.n_features_ = None

    def bootstrap_sample(self, X, y, rng):
        n_samples = X.shape[0]
        idx = rng.choice(n_samples, n_samples, replace=True)
        return X[idx], y[idx]

    def _gini_importance(self, tree):
        if tree["type"] == "leaf":
            return {}
        fid = tree["feature"]
        samples = tree["samples"]
        imp = {fid: samples * tree["gain"]}
        imp_left = self._gini_importance(tree["left"])
        imp_right = self._gini_importance(tree["right"])
        for k, v in imp_left.items():
            imp[k] = imp.get(k, 0) + v
        for k, v in imp_right.items():
            imp[k] = imp.get(k, 0) + v
        return imp

    def fit(self, X, y, feature_names=None):
        n_samples, n_features = X.shape
        self.n_features_ = n_features

        if self.max_features is None:
            self.max_features = max(1, int(np.sqrt(n_features)))

        self.trees = []
        gini_imp_total = {}
        rng = np.random.default_rng(self.random_state)

        for i in range(self.n_estimators):
            X_sample, y_sample = self.bootstrap_sample(X, y, rng)

            feat_idx = rng.choice(
                n_features, self.max_features, replace=False
            )
            X_sample_sub = X_sample[:, feat_idx]

            tree = build_cart_tree(
                X_sample_sub, y_sample,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
            )

            self.trees.append((tree, feat_idx))

            tree_imp = self._gini_importance(tree)
            for local_fid, v in tree_imp.items():
                global_fid = feat_idx[local_fid]
                gini_imp_total[global_fid] = gini_imp_total.get(global_fid, 0) + v

        importances = np.zeros(n_features)
        for fid, v in gini_imp_total.items():
            importances[fid] = v
        total = importances.sum()
        if total > 0:
            importances = importances / total
        self.feature_importances_ = importances

    def predict_one(self, x):
        preds = []
        for tree, feat_idx in self.trees:
            x_sub = x[feat_idx]
            pred = predict_one(tree, x_sub)
            preds.append(pred)
        return Counter(preds).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X], dtype=int)

    def predict_proba(self, X):
        results = []
        for x in X:
            preds = []
            for tree, feat_idx in self.trees:
                x_sub = x[feat_idx]
                pred = predict_one(tree, x_sub)
                preds.append(pred)
            counter = Counter(preds)
            total = len(preds)
            prob = {k: v / total for k, v in counter.items()}
            results.append(prob)
        return results


def plot_feature_importance(importances, feature_names, save_path):
    from pathlib import Path
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    indices = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in indices]
    sorted_vals = importances[indices]

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(sorted_vals)))

    plt.figure(figsize=(10, 6))
    bars = plt.barh(range(len(sorted_vals)), sorted_vals, color=colors[::-1])
    plt.yticks(range(len(sorted_vals)), sorted_names)
    plt.xlabel("Feature Importance")
    plt.title("Feature Importance of Random Forest")
    plt.gca().invert_yaxis()

    for bar, val in zip(bars, sorted_vals):
        plt.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[已保存] 特征重要性图：{save_path}")


def plot_estimator_comparison(
    X_train, y_train, X_test, y_test,
    estimator_range, save_path, rf_kwargs=None,
):
    from pathlib import Path
    from sklearn.metrics import accuracy_score
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if rf_kwargs is None:
        rf_kwargs = {}

    train_accs = []
    test_accs = []

    for n in estimator_range:
        rf = RandomForest(n_estimators=n, random_state=42, **rf_kwargs)
        rf.fit(X_train, y_train)
        train_accs.append(accuracy_score(y_train, rf.predict(X_train)))
        test_accs.append(accuracy_score(y_test, rf.predict(X_test)))

    plt.figure(figsize=(8, 5))
    plt.plot(estimator_range, train_accs, marker="o", label="Train Accuracy")
    plt.plot(estimator_range, test_accs, marker="s", label="Test Accuracy")
    plt.title("RF Performance by n_estimators")
    plt.xlabel("n_estimators")
    plt.ylabel("Accuracy")
    plt.xticks(estimator_range)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[已保存] 树数量对比图：{save_path}")
