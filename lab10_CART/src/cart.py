# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Noto Sans CJK JP", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


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
            X_left,
            y_left,
            depth + 1,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            min_gain,
        ),
        "right": build_cart_tree(
            X_right,
            y_right,
            depth + 1,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            min_gain,
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


def _assign_positions(tree, depth, positions, counter):
    node_id = id(tree)

    if tree["type"] == "leaf":
        x = counter[0]
        counter[0] += 1
        y = -depth
        positions[node_id] = (x, y)
        return x

    left_x = _assign_positions(tree["left"], depth + 1, positions, counter)
    right_x = _assign_positions(tree["right"], depth + 1, positions, counter)

    x = (left_x + right_x) / 2
    y = -depth
    positions[node_id] = (x, y)

    return x


def _draw_tree(ax, tree, positions, feature_names, class_names):
    x, y = positions[id(tree)]

    if tree["type"] == "leaf":
        label = (
            f"{class_names.get(tree['class'], tree['class'])}\n"
            f"samples={tree['samples']}\n"
            f"gini={tree['gini']:.3f}"
        )
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", fc="lightgreen", ec="green"),
        )
        return

    feature = feature_names[tree["feature"]]
    label = (
        f"{feature} <= {tree['threshold']}\n"
        f"samples={tree['samples']}\n"
        f"gini={tree['gini']:.3f}\n"
        f"gain={tree['gain']:.3f}"
    )

    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="lightblue", ec="blue"),
    )

    for child_key, edge_text in [("left", "是"), ("right", "否")]:
        child = tree[child_key]
        child_x, child_y = positions[id(child)]

        ax.plot([x, child_x], [y - 0.15, child_y + 0.15], linewidth=1)

        ax.text(
            (x + child_x) / 2,
            (y + child_y) / 2,
            edge_text,
            ha="center",
            va="center",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="gray"),
        )

        _draw_tree(ax, child, positions, feature_names, class_names)


def plot_tree(tree, feature_names, class_names, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    positions = {}
    _assign_positions(tree, 0, positions, [0])

    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]

    width = max(10, len(xs) * 1.8)
    height = max(6, abs(min(ys)) * 1.8 + 2)

    fig, ax = plt.subplots(figsize=(width, height))
    _draw_tree(ax, tree, positions, feature_names, class_names)

    ax.set_xlim(min(xs) - 1, max(xs) + 1)
    ax.set_ylim(min(ys) - 1, 1)
    ax.axis("off")
    ax.set_title("CART Decision Tree")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[已保存] CART 决策树图片：{save_path}")
