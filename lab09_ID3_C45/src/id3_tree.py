import numpy as np


class ID3Tree:
    def __init__(self):
        self.tree = None
        self.majority_class = None

    def calculate_entropy(self, y):
        """
        计算信息熵。
        熵越小，说明用户类别越集中。
        """
        classes = np.unique(y)
        entropy = 0.0

        for cls in classes:
            p = np.sum(y == cls) / len(y)
            if p > 0:
                entropy -= p * np.log2(p)

        return entropy

    def calculate_information_gain(self, X, y, feature):
        """
        计算某个特征的信息增益。
        ID3 选择信息增益最大的特征作为分裂特征。
        """
        total_entropy = self.calculate_entropy(y)
        values = X[feature].unique()

        conditional_entropy = 0.0
        for value in values:
            mask = X[feature] == value
            subset_y = y[mask]
            weight = len(subset_y) / len(y)
            conditional_entropy += weight * self.calculate_entropy(subset_y)

        return total_entropy - conditional_entropy

    def majority_vote(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def build_tree(self, X, y, features):
        """
        递归构建 ID3 决策树。
        """
        if len(np.unique(y)) == 1:
            return int(np.unique(y)[0])

        if len(features) == 0:
            return int(self.majority_vote(y))

        gains = [self.calculate_information_gain(X, y, feature) for feature in features]
        best_feature = features[int(np.argmax(gains))]

        tree = {best_feature: {}}
        remaining_features = [feature for feature in features if feature != best_feature]

        for value in X[best_feature].unique():
            mask = X[best_feature] == value
            subset_X = X[mask]
            subset_y = y[mask]

            if len(subset_y) == 0:
                tree[best_feature][value] = int(self.majority_vote(y))
            else:
                tree[best_feature][value] = self.build_tree(
                    subset_X,
                    subset_y,
                    remaining_features
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

        feature = list(tree.keys())[0]
        value = sample[feature]

        if value in tree[feature]:
            return self.predict_one(sample, tree[feature][value])

        return self.majority_class

    def predict(self, X):
        return np.array([self.predict_one(row) for _, row in X.iterrows()])
