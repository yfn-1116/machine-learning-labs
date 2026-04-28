"""
机器学习实验 1：Scikit-Learn的学习与运用（聚类算法实践）
实验脚本 - 包含完整的聚类分析流程
修复问题：将图表标签改为英文以解决云端环境中文显示为方框的问题
"""

# 设置matplotlib后端为Agg，避免GUI依赖
import matplotlib

matplotlib.use("Agg")
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score

# 输出目录为脚本所在目录
OUTPUT_DIR = os.path.dirname(__file__)
if OUTPUT_DIR == "":
    OUTPUT_DIR = ".."  # 防止空路径
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

# 确保保存目录存在
if not os.path.exists(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)

# 设置可视化样式
# 注意：移除了 SimHei 设置，因为云端通常没有该字体，改用默认字体配合英文
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")

print("=" * 80)
print("机器学习实验 1：Scikit-Learn的学习与运用（聚类算法实践）")
print("=" * 80)

# ====================== 1. 数据集准备 ======================
print("\n1. 数据集准备")
print("-" * 40)

# 1.1 生成模拟数据集（make_blobs）
print("生成模拟数据集...")
X_blobs, y_blobs_true = make_blobs(
    n_samples=500, n_features=2, centers=3, cluster_std=1.0, random_state=42
)

# 标准化
scaler = StandardScaler()
X_blobs_scaled = scaler.fit_transform(X_blobs)
print(f"模拟数据集形状: {X_blobs.shape} -> {X_blobs_scaled.shape}")

# 1.2 加载Iris数据集
print("\n加载Iris数据集...")
iris = load_iris()
X_iris = iris.data
y_iris_true = iris.target
X_iris_scaled = scaler.fit_transform(X_iris)

# PCA降维
pca = PCA(n_components=2, random_state=42)
X_iris_pca = pca.fit_transform(X_iris_scaled)
print(f"Iris数据集形状: {X_iris.shape} -> {X_iris_pca.shape}")

# ====================== 2. 数据可视化 (英文标签) ======================
print("\n2. 数据可视化")
print("-" * 40)

# 2.1 模拟数据集原始分布
plt.figure(figsize=(8, 6))
plt.scatter(
    X_blobs_scaled[:, 0], X_blobs_scaled[:, 1], c="gray", alpha=0.6, edgecolors="k"
)
# 修改：使用英文标题
plt.title("Simulated Data Raw Distribution (Standardized)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True, alpha=0.3)
plt.savefig(
    os.path.join(FIGURES_DIR, "blobs_raw_distribution.png"),
    dpi=150,
    bbox_inches="tight",
)
plt.close()
print("已保存: blobs_raw_distribution.png")

# 2.2 Iris数据集PCA降维后分布
plt.figure(figsize=(8, 6))
plt.scatter(X_iris_pca[:, 0], X_iris_pca[:, 1], c="gray", alpha=0.6, edgecolors="k")
# 修改：使用英文标题
plt.title("Iris Data PCA Distribution (Standardized)")
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.grid(True, alpha=0.3)
plt.savefig(
    os.path.join(FIGURES_DIR, "iris_pca_distribution.png"), dpi=150, bbox_inches="tight"
)
plt.close()
print("已保存: iris_pca_distribution.png")

# ====================== 3. K-Means模型构建与训练 ======================
print("\n3. K-Means模型构建与训练")
print("-" * 40)

# 3.1 模拟数据集训练
kmeans_blobs = KMeans(n_clusters=3, random_state=42, max_iter=300, n_init=10)
kmeans_blobs.fit(X_blobs_scaled)
y_blobs_pred = kmeans_blobs.labels_
centers_blobs = kmeans_blobs.cluster_centers_
inertia_blobs = kmeans_blobs.inertia_

print("=== 模拟数据集K-Means聚类结果 ===")
print(f"簇内平方和（Inertia）: {inertia_blobs:.2f}")

# 3.2 Iris数据集训练
kmeans_iris = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_iris.fit(X_iris_scaled)
y_iris_pred = kmeans_iris.labels_
centers_iris = kmeans_iris.cluster_centers_
inertia_iris = kmeans_iris.inertia_
centers_iris_pca = pca.transform(centers_iris)

print("=== Iris数据集K-Means聚类结果 ===")
print(f"簇内平方和（Inertia）: {inertia_iris:.2f}")

# ====================== 4. 聚类结果可视化 (英文标签) ======================
print("\n4. 聚类结果可视化")
print("-" * 40)

# 4.1 模拟数据集聚类结果
plt.figure(figsize=(8, 6))
plt.scatter(
    X_blobs_scaled[:, 0],
    X_blobs_scaled[:, 1],
    c=y_blobs_pred,
    cmap="viridis",
    alpha=0.6,
    edgecolors="k",
)
plt.scatter(
    centers_blobs[:, 0],
    centers_blobs[:, 1],
    c="red",
    marker="*",
    s=200,
    edgecolors="k",
    label="Centroids",
)
# 修改：使用英文标题
plt.title("Simulated Data K-Means Result (K=3)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(
    os.path.join(FIGURES_DIR, "blobs_kmeans_result.png"), dpi=150, bbox_inches="tight"
)
plt.close()
print("已保存: blobs_kmeans_result.png")

# 4.2 Iris数据集聚类结果
plt.figure(figsize=(8, 6))
plt.scatter(
    X_iris_pca[:, 0],
    X_iris_pca[:, 1],
    c=y_iris_pred,
    cmap="viridis",
    alpha=0.6,
    edgecolors="k",
)
plt.scatter(
    centers_iris_pca[:, 0],
    centers_iris_pca[:, 1],
    c="red",
    marker="*",
    s=200,
    edgecolors="k",
    label="Centroids",
)
# 修改：使用英文标题
plt.title("Iris Data K-Means Result (PCA Reduced, K=3)")
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(
    os.path.join(FIGURES_DIR, "iris_kmeans_result.png"), dpi=150, bbox_inches="tight"
)
plt.close()
print("已保存: iris_kmeans_result.png")

# 4.3 对比Iris真实标签与聚类结果
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
# 真实标签
ax1.scatter(
    X_iris_pca[:, 0],
    X_iris_pca[:, 1],
    c=y_iris_true,
    cmap="viridis",
    alpha=0.6,
    edgecolors="k",
)
ax1.set_title("Iris Ground Truth")  # 英文标题
ax1.set_xlabel("PCA Dimension 1")
ax1.set_ylabel("PCA Dimension 2")
ax1.grid(True, alpha=0.3)
# 聚类结果
ax2.scatter(
    X_iris_pca[:, 0],
    X_iris_pca[:, 1],
    c=y_iris_pred,
    cmap="viridis",
    alpha=0.6,
    edgecolors="k",
)
ax2.set_title("Iris K-Means Prediction (K=3)")  # 英文标题
ax2.set_xlabel("PCA Dimension 1")
ax2.set_ylabel("PCA Dimension 2")
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(FIGURES_DIR, "iris_real_vs_pred.png"), dpi=150, bbox_inches="tight"
)
plt.close()
print("已保存: iris_real_vs_pred.png")

# ====================== 5. 聚类质量评估 ======================
print("\n5. 聚类质量评估")
sil_blobs = silhouette_score(X_blobs_scaled, y_blobs_pred)
sil_iris = silhouette_score(X_iris_scaled, y_iris_pred)
ari_iris = adjusted_rand_score(y_iris_true, y_iris_pred)
print(f"模拟数据集 Silhouette: {sil_blobs:.4f}")
print(f"Iris数据集 Silhouette: {sil_iris:.4f}")
print(f"Iris数据集 ARI: {ari_iris:.4f}")

# ====================== 6. 最佳簇数K的确定 (英文标签) ======================
print("\n6. 最佳簇数K的确定")
k_range = range(1, 11)

# 模拟数据集计算
inertias_blobs = []
sil_scores_blobs = []
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_blobs_scaled)
    inertias_blobs.append(kmeans.inertia_)
    if k >= 2:
        sil_scores_blobs.append(silhouette_score(X_blobs_scaled, kmeans.labels_))
    else:
        sil_scores_blobs.append(np.nan)

# Iris数据集计算
inertias_iris = []
sil_scores_iris = []
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_iris_scaled)
    inertias_iris.append(kmeans.inertia_)
    if k >= 2:
        sil_scores_iris.append(silhouette_score(X_iris_scaled, kmeans.labels_))
    else:
        sil_scores_iris.append(np.nan)

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Blobs - Elbow
axes[0, 0].plot(k_range, inertias_blobs, marker="o", linestyle="-", color="blue")
axes[0, 0].axvline(x=3, color="red", linestyle="--", label="Best K=3")
axes[0, 0].set_title("Blobs: Elbow Method (Inertia vs K)")
axes[0, 0].set_xlabel("Number of Clusters K")
axes[0, 0].set_ylabel("Inertia")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Blobs - Silhouette
axes[0, 1].plot(
    k_range[1:], sil_scores_blobs[1:], marker="o", linestyle="-", color="green"
)
axes[0, 1].axvline(x=3, color="red", linestyle="--", label="Best K=3")
axes[0, 1].set_title("Blobs: Silhouette Score vs K")
axes[0, 1].set_xlabel("Number of Clusters K")
axes[0, 1].set_ylabel("Silhouette Score")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Iris - Elbow
axes[1, 0].plot(k_range, inertias_iris, marker="o", linestyle="-", color="blue")
axes[1, 0].axvline(x=3, color="red", linestyle="--", label="Best K=3")
axes[1, 0].set_title("Iris: Elbow Method (Inertia vs K)")
axes[1, 0].set_xlabel("Number of Clusters K")
axes[1, 0].set_ylabel("Inertia")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Iris - Silhouette
axes[1, 1].plot(
    k_range[1:], sil_scores_iris[1:], marker="o", linestyle="-", color="green"
)
axes[1, 1].axvline(x=3, color="red", linestyle="--", label="Best K=3")
axes[1, 1].set_title("Iris: Silhouette Score vs K")
axes[1, 1].set_xlabel("Number of Clusters K")
axes[1, 1].set_ylabel("Silhouette Score")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(FIGURES_DIR, "elbow_silhouette_curves.png"),
    dpi=150,
    bbox_inches="tight",
)
plt.close()
print("已保存: elbow_silhouette_curves.png")

# ====================== 7. 其他聚类算法对比 (英文标签) ======================
print("\n7. 其他聚类算法对比")

# 层次聚类
agg_clust = AgglomerativeClustering(n_clusters=3, linkage="ward")
y_agg_pred = agg_clust.fit_predict(X_blobs_scaled)
sil_agg = silhouette_score(X_blobs_scaled, y_agg_pred)

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
y_dbscan_pred = dbscan.fit_predict(X_blobs_scaled)

# 处理DBSCAN噪声用于计算分数
mask = y_dbscan_pred != -1
if np.sum(mask) > 1:
    sil_dbscan = silhouette_score(X_blobs_scaled[mask], y_dbscan_pred[mask])
else:
    sil_dbscan = 0

# 可视化对比
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# K-Means
ax1.scatter(
    X_blobs_scaled[:, 0],
    X_blobs_scaled[:, 1],
    c=y_blobs_pred,
    cmap="viridis",
    alpha=0.6,
    edgecolors="k",
)
ax1.set_title(f"K-Means (K=3, Sil={sil_blobs:.3f})")
ax1.set_xlabel("Feature 1")
ax1.set_ylabel("Feature 2")
ax1.grid(True, alpha=0.3)

# 层次聚类
ax2.scatter(
    X_blobs_scaled[:, 0],
    X_blobs_scaled[:, 1],
    c=y_agg_pred,
    cmap="viridis",
    alpha=0.6,
    edgecolors="k",
)
ax2.set_title(f"Agglomerative (K=3, Sil={sil_agg:.3f})")
ax2.set_xlabel("Feature 1")
ax2.set_ylabel("Feature 2")
ax2.grid(True, alpha=0.3)

# DBSCAN
ax3.scatter(
    X_blobs_scaled[:, 0],
    X_blobs_scaled[:, 1],
    c=y_dbscan_pred,
    cmap="viridis",
    alpha=0.6,
    edgecolors="k",
)
ax3.set_title(f"DBSCAN (eps=0.5, Sil={sil_dbscan:.3f})")
ax3.set_xlabel("Feature 1")
ax3.set_ylabel("Feature 2")
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(FIGURES_DIR, "algorithm_comparison.png"), dpi=150, bbox_inches="tight"
)
plt.close()
print("已保存: algorithm_comparison.png")

print("\n实验完成！请查看 figures 文件夹下的图片，现在文字应显示正常。")
