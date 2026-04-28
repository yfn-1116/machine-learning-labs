import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
)


class SimpleAGNES:
    """
    简化版 AGNES，仅用于原理验证
    """

    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters
        self.labels_ = None

    def fit(self, X):
        n_samples = X.shape[0]
        clusters = [[i] for i in range(n_samples)]
        self.labels_ = np.zeros(n_samples, dtype=int)

        while len(clusters) > self.n_clusters:
            cluster_centers = [np.mean(X[cluster], axis=0) for cluster in clusters]
            dist_matrix = np.zeros((len(clusters), len(clusters)))

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                    dist_matrix[i, j] = dist
                    dist_matrix[j, i] = dist

            np.fill_diagonal(dist_matrix, np.inf)
            i, j = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)

            new_cluster = clusters[i] + clusters[j]
            clusters = [clusters[k] for k in range(len(clusters)) if k != i and k != j]
            clusters.append(new_cluster)

        for label, cluster in enumerate(clusters):
            self.labels_[cluster] = label

        return self


def run_agnes(X_scaled, n_clusters=4, linkage_method="ward"):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = model.fit_predict(X_scaled)
    return model, labels


def compare_linkage_methods(X_scaled, linkage_methods=None, n_clusters=4):
    if linkage_methods is None:
        linkage_methods = ["ward", "average", "complete", "single"]

    results = {}
    for method in linkage_methods:
        model, labels = run_agnes(
            X_scaled, n_clusters=n_clusters, linkage_method=method
        )
        results[method] = {
            "model": model,
            "labels": labels,
        }
    return results


def calculate_wcss(X, max_clusters=10, linkage_method="ward"):
    cluster_counts = list(range(2, max_clusters + 1))
    wcss_values = []

    for n in cluster_counts:
        _, labels = run_agnes(X, n_clusters=n, linkage_method=linkage_method)

        wcss = 0.0
        for i in range(n):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                centroid = np.mean(cluster_points, axis=0)
                wcss += np.sum((cluster_points - centroid) ** 2)

        wcss_values.append(wcss)

    return cluster_counts, wcss_values


def calculate_silhouette(X, max_clusters=10, linkage_method="ward"):
    cluster_counts = list(range(2, max_clusters + 1))
    silhouette_values = []

    for n in cluster_counts:
        _, labels = run_agnes(X, n_clusters=n, linkage_method=linkage_method)
        sil = silhouette_score(X, labels)
        silhouette_values.append(sil)

    return cluster_counts, silhouette_values


def dunn_index(X, labels):
    cluster_ids = np.unique(labels)

    max_diameter = 0.0
    for cid in cluster_ids:
        cluster_points = X[labels == cid]
        if len(cluster_points) > 1:
            intra_dists = pdist(cluster_points)
            if len(intra_dists) > 0:
                max_diameter = max(max_diameter, np.max(intra_dists))

    if max_diameter == 0:
        return 0.0

    min_inter_dist = np.inf
    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            points_i = X[labels == cluster_ids[i]]
            points_j = X[labels == cluster_ids[j]]

            for pi in points_i:
                for pj in points_j:
                    dist = np.linalg.norm(pi - pj)
                    if dist < min_inter_dist:
                        min_inter_dist = dist

    if min_inter_dist == np.inf:
        return 0.0

    return min_inter_dist / max_diameter


def evaluate_internal_metrics(X_scaled, linkage_methods, n_clusters=4):
    rows = []

    for method in linkage_methods:
        _, labels = run_agnes(X_scaled, n_clusters=n_clusters, linkage_method=method)

        rows.append(
            {
                "Linkage": method,
                "轮廓系数": silhouette_score(X_scaled, labels),
                "CH指数": calinski_harabasz_score(X_scaled, labels),
                "Dunn指数": dunn_index(X_scaled, labels),
            }
        )

    return pd.DataFrame(rows)


def evaluate_external_metrics(X_scaled, y_true, linkage_methods, n_clusters=4):
    rows = []

    for method in linkage_methods:
        _, labels = run_agnes(X_scaled, n_clusters=n_clusters, linkage_method=method)

        rows.append(
            {
                "Linkage": method,
                "ARI": adjusted_rand_score(y_true, labels),
                "同质性": homogeneity_score(y_true, labels),
                "完整性": completeness_score(y_true, labels),
                "V-measure": v_measure_score(y_true, labels),
            }
        )

    return pd.DataFrame(rows)


def build_dendrogram_linkage(X_scaled, method="ward"):
    return linkage(X_scaled, method=method)
