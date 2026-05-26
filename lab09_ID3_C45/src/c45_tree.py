import numpy as np
import pandas as pd


class C45Tree:
    def __init__(self, min_samples=5, max_depth=5, min_gain_ratio=0.01):
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.min_gain_ratio = min_gain_ratio
        self.tree = None
        self.majority_class = None

    def calculate_entropy(self, y):
        classes = np.unique(y)
        entropy = 0.0

        for cls in classes:
            p = np.sum(y == cls) / len(y)
            if p > 0:
                entropy -= p * np.log2(p)

        return entropy

    def majority_vote(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def information_gain_discrete(self, X, y, feature):
        non_missing = ~X[feature].isnull()
        X_valid = X[non_missing]
        y_valid = y[non_missing]

        if len(y_valid) == 0:
            return 0

        total_entropy = self.calculate_entropy(y_valid)
        values = X_valid[feature].unique()

        conditional_entropy = 0.0
        split_info = 0.0

        for value in values:
            mask = X_valid[feature] == value
            subset_y = y_valid[mask]
            p = len(subset_y) / len(y_valid)

            conditional_entropy += p * self.calculate_entropy(subset_y)

            if p > 0:
                split_info -= p * np.log2(p)

        gain = total_entropy - conditional_entropy

        if split_info == 0:
            return 0

        return gain / split_info

    def best_split_continuous(self, X, y, feature):
        non_missing = ~X[feature].isnull()
        X_valid = X[non_missing]
        y_valid = y[non_missing]

        values = np.sort(X_valid[feature].unique())

        if len(values) <= 1:
            return None, 0

        best_split = None
        best_gain_ratio = 0

        total_entropy = self.calculate_entropy(y_valid)

        for i in range(len(values) - 1):
            split = (values[i] + values[i + 1]) / 2

            left_y = y_valid[X_valid[feature] <= split]
            right_y = y_valid[X_valid[feature] > split]

            if len(left_y) == 0 or len(right_y) == 0:
                continue

            p_left = len(left_y) / len(y_valid)
            p_right = len(right_y) / len(y_valid)

            conditional_entropy = (
                p_left * self.calculate_entropy(left_y)
                + p_right * self.calculate_entropy(right_y)
            )

            gain = total_entropy - conditional_entropy

            split_info = 0
            for p in [p_left, p_right]:
                if p > 0:
                    split_info -= p * np.log2(p)

            if split_info == 0:
                continue

            gain_ratio = gain / split_info

            if gain_ratio > best_gain_ratio:
                best_gain_ratio = gain_ratio
                best_split = split

        return best_split, best_gain_ratio

    def choose_best_feature(self, X, y, features):
        best_feature = None
        best_split = None
        best_gain_ratio = 0
        best_is_continuous = False

        for feature in features:
            if X[feature].dtype in ["int64", "float64"] and X[feature].nunique() > 10:
                split, gain_ratio = self.best_split_continuous(X, y, feature)
                is_continuous = True
            else:
                split = None
                gain_ratio = self.information_gain_discrete(X, y, feature)
                is_continuous = False

            if gain_ratio > best_gain_ratio:
                best_gain_ratio = gain_ratio
                best_feature = feature
                best_split = split
                best_is_continuous = is_continuous

        return best_feature, best_split, best_gain_ratio, best_is_continuous

    def build_tree(self, X, y, features, depth=0):
        if len(np.unique(y)) == 1:
            return int(np.unique(y)[0])

        if len(features) == 0:
            return int(self.majority_vote(y))

        if len(y) < self.min_samples or depth >= self.max_depth:
            return int(self.majority_vote(y))

        best_feature, best_split, best_gain_ratio, is_continuous = self.choose_best_feature(
            X, y, features
        )

        if best_feature is None or best_gain_ratio < self.min_gain_ratio:
            return int(self.majority_vote(y))

        if is_continuous:
            node_name = f"{best_feature} <= {best_split:.4f}"
            tree = {node_name: {}}

            left_mask = X[best_feature] <= best_split
            right_mask = X[best_feature] > best_split
            missing_mask = X[best_feature].isnull()

            tree[node_name]["是"] = self.build_tree(
                X[left_mask | missing_mask],
                y[left_mask | missing_mask],
                features,
                depth + 1
            )

            tree[node_name]["否"] = self.build_tree(
                X[right_mask | missing_mask],
                y[right_mask | missing_mask],
                features,
                depth + 1
            )

            return tree

        tree = {best_feature: {}}
        remaining_features = [feature for feature in features if feature != best_feature]

        for value in X[best_feature].dropna().unique():
            mask = X[best_feature] == value
            missing_mask = X[best_feature].isnull()

            tree[best_feature][value] = self.build_tree(
                X[mask | missing_mask],
                y[mask | missing_mask],
                remaining_features,
                depth + 1
            )

        return tree

    def fit(self, X, y):
        self.majority_class = int(self.majority_vote(y))
        self.tree = self.build_tree(X, y, list(X.columns))
        return self

    def predict_one(self, sample, tree=None):
        if tree is None:
            tree = self.tree

        if not isinstance(tree, dict):
            return int(tree)

        node = list(tree.keys())[0]

        if "<=" in node:
            feature, split = node.split(" <= ")
            split = float(split)

            value = sample[feature]

            if pd.isnull(value):
                branch = list(tree[node].keys())[0]
            elif value <= split:
                branch = "是"
            else:
                branch = "否"

            return self.predict_one(sample, tree[node][branch])

        feature = node
        value = sample[feature]

        if value in tree[feature]:
            return self.predict_one(sample, tree[feature][value])

        return self.majority_class

    def predict(self, X):
        return np.array([self.predict_one(row) for _, row in X.iterrows()])
