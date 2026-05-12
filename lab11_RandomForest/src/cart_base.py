# -*- coding: utf-8 -*-
import numpy as np


def gini(y):
    y = np.asarray(y)
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1.0 - np.sum(p ** 2)


def majority_class(y):
    labels, counts = np.unique(y, return_counts=True)
    return int(labels[np.argmax(counts)])


def split_dataset(X, y, feature_idx, threshold):
    mask = X[:, feature_idx] <= threshold
    return X[mask], y[mask], X[~mask], y[~mask]


def best_split(X, y, min_samples_leaf=2):
    n_samples, n_features = X.shape
    current_gini = gini(y)

    best_feature = None
    best_threshold = None
    best_gain = 0.0

    for fid in range(n_features):
        values = X[:, fid]
        unique_values = np.unique(values)

        if len(unique_values) <= 1:
            continue

        thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0

        for t in thresholds:
            X_left, y_left, X_right, y_right = split_dataset(X, y, fid, t)

            if len(y_left) < min_samples_leaf or len(y_right) < min_samples_leaf:
                continue

            new_gini = len(y_left) / len(y) * gini(y_left) + len(y_right) / len(y) * gini(y_right)
            gain = current_gini - new_gini

            if gain > best_gain:
                best_gain = gain
                best_feature = fid
                best_threshold = float(t)

    return best_feature, best_threshold, best_gain


def build_cart_tree(
    X,
    y,
    depth=0,
    max_depth=4,
    min_samples_split=5,
    min_samples_leaf=2,
    min_gain=1e-6,
):
    y = np.asarray(y).astype(int)

    node_gini = gini(y)
    node_majority = majority_class(y)

    if len(np.unique(y)) == 1:
        return {
            "type": "leaf",
            "class": int(y[0]),
            "samples": len(y),
            "gini": node_gini,
        }

    if depth >= max_depth or len(y) < min_samples_split:
        return {
            "type": "leaf",
            "class": node_majority,
            "samples": len(y),
            "gini": node_gini,
        }

    fid, threshold, gain = best_split(X, y, min_samples_leaf=min_samples_leaf)

    if fid is None or gain < min_gain:
        return {
            "type": "leaf",
            "class": node_majority,
            "samples": len(y),
            "gini": node_gini,
        }

    X_left, y_left, X_right, y_right = split_dataset(X, y, fid, threshold)

    if len(y_left) < min_samples_leaf or len(y_right) < min_samples_leaf:
        return {
            "type": "leaf",
            "class": node_majority,
            "samples": len(y),
            "gini": node_gini,
        }

    return {
        "type": "node",
        "feature": int(fid),
        "threshold": round(float(threshold), 4),
        "gain": float(gain),
        "gini": node_gini,
        "samples": len(y),
        "left": build_cart_tree(
            X_left, y_left, depth + 1,
            max_depth, min_samples_split, min_samples_leaf, min_gain,
        ),
        "right": build_cart_tree(
            X_right, y_right, depth + 1,
            max_depth, min_samples_split, min_samples_leaf, min_gain,
        ),
    }


def predict_one(tree, x):
    if tree["type"] == "leaf":
        return int(tree["class"])

    fid = tree["feature"]
    threshold = tree["threshold"]

    if x[fid] <= threshold:
        return predict_one(tree["left"], x)
    else:
        return predict_one(tree["right"], x)


def predict(tree, X):
    return np.array([predict_one(tree, row) for row in X], dtype=int)


def tree_depth(tree):
    if tree["type"] == "leaf":
        return 1
    return 1 + max(tree_depth(tree["left"]), tree_depth(tree["right"]))


def count_leaves(tree):
    if tree["type"] == "leaf":
        return 1
    return count_leaves(tree["left"]) + count_leaves(tree["right"])


def format_rules(tree, feature_names, class_names, depth=0):
    lines = []
    indent = "    " * depth

    if tree["type"] == "leaf":
        class_name = class_names.get(tree["class"], str(tree["class"]))
        lines.append(
            f"{indent}→ 判定：{class_name} "
            f"(样本数={tree['samples']}, Gini={tree['gini']:.4f})"
        )
        return lines

    feature = feature_names[tree["feature"]]
    threshold = tree["threshold"]

    lines.append(
        f"{indent}若 {feature} <= {threshold} "
        f"(样本数={tree['samples']}, Gini={tree['gini']:.4f}, Gain={tree['gain']:.4f})"
    )
    lines.extend(format_rules(tree["left"], feature_names, class_names, depth + 1))
    lines.append(f"{indent}若 {feature} > {threshold}")
    lines.extend(format_rules(tree["right"], feature_names, class_names, depth + 1))

    return lines


def print_rules(tree, feature_names, class_names):
    text = "\n".join(format_rules(tree, feature_names, class_names))
    print(text)
    return text
