import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram

from datasets import generate_user_data, remove_outliers, standardize_features
from agnes import (
    SimpleAGNES,
    compare_linkage_methods,
    calculate_wcss,
    calculate_silhouette,
    evaluate_internal_metrics,
    evaluate_external_metrics,
    run_agnes,
    build_dendrogram_linkage,
)

# Prevent minus sign display issues
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_plot(filename):
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close()


def main():
    # 1. Generate data
    df = generate_user_data(random_state=42, n_samples=400)

    feature_cols = ["reg_days", "browse_duration", "order_count", "avg_order_value"]

    # 2. Remove outliers
    df = remove_outliers(df, feature_cols, q=0.99)

    # 3. Standardize features
    X_scaled, scaler = standardize_features(df, feature_cols)
    y_true = df["true_label"].values

    # Save simulated data
    df.to_csv(
        os.path.join(OUTPUT_DIR, "00_simulated_user_data.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # 4. Simple AGNES demonstration
    simple_agnes = SimpleAGNES(n_clusters=4)
    simple_agnes.fit(X_scaled[:30])

    print("=== Simple AGNES labels for first 30 samples ===")
    print(simple_agnes.labels_)

    # 5. Compare linkage methods
    linkage_methods = ["ward", "average", "complete", "single"]
    results = compare_linkage_methods(
        X_scaled, linkage_methods=linkage_methods, n_clusters=4
    )

    print("\n=== Cluster label distribution for different linkage methods ===")
    for method, result in results.items():
        print(method, pd.Series(result["labels"]).value_counts().sort_index().to_dict())

    # 6. Dendrogram
    Z = build_dendrogram_linkage(X_scaled, method="ward")
    plt.figure(figsize=(12, 7))
    dendrogram(Z, truncate_mode="level", p=5, leaf_rotation=45, leaf_font_size=10)
    plt.title("AGNES Dendrogram (Ward Linkage)")
    plt.xlabel("Sample Index (or Cluster Index)")
    plt.ylabel("Merge Distance")
    plt.grid(True, alpha=0.3, axis="y")
    save_plot("01_dendrogram_ward.png")

    # 7. Elbow method
    cluster_counts, wcss_values = calculate_wcss(
        X_scaled, max_clusters=10, linkage_method="ward"
    )
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_counts, wcss_values, "bo-", linewidth=2, markersize=8)
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
    plt.title("Elbow Method for Optimal Cluster Number (Ward Linkage)")
    plt.grid(True, alpha=0.3)
    save_plot("02_elbow_wcss.png")

    # 8. Silhouette method
    cluster_counts_sil, silhouette_values = calculate_silhouette(
        X_scaled, max_clusters=10, linkage_method="ward"
    )

    plt.figure(figsize=(10, 6))
    plt.plot(cluster_counts_sil, silhouette_values, "go-", linewidth=2, markersize=8)
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Average Silhouette Score")
    plt.title("Silhouette Method for Optimal Cluster Number (Ward Linkage)")
    plt.grid(True, alpha=0.3)

    best_sil_idx = np.argmax(silhouette_values)
    plt.axvline(
        x=cluster_counts_sil[best_sil_idx],
        color="red",
        linestyle="--",
        label=f"Best k = {cluster_counts_sil[best_sil_idx]}",
    )
    plt.legend()
    save_plot("03_silhouette_k.png")

    # Fixed best cluster number
    n_clusters_best = 4

    # 9. Internal metrics
    df_internal = evaluate_internal_metrics(
        X_scaled, linkage_methods, n_clusters=n_clusters_best
    )
    df_internal.to_csv(
        os.path.join(OUTPUT_DIR, "04_internal_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== Internal Metrics ===")
    print(df_internal)

    # 10. External metrics
    df_external = evaluate_external_metrics(
        X_scaled, y_true, linkage_methods, n_clusters=n_clusters_best
    )
    df_external.to_csv(
        os.path.join(OUTPUT_DIR, "05_external_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== External Metrics ===")
    print(df_external)

    # 11. Final model
    _, final_labels = run_agnes(X_scaled, n_clusters=4, linkage_method="ward")
    df["cluster_label"] = final_labels

    # 12. PCA visualization
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10, 6))
    for i in range(4):
        mask = df["cluster_label"] == i
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], s=50, alpha=0.7, label=f"Cluster {i}")

    plt.title("AGNES Clustering Result (Ward Linkage, PCA Projection)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot("06_final_clusters_pca.png")

    # 13. Cluster profiling
    cluster_means = df.groupby("cluster_label")[feature_cols].mean()
    cluster_counts = df["cluster_label"].value_counts().sort_index()
    cluster_means["sample_count"] = cluster_counts

    cluster_names = {
        0: "Newly Registered Observers",
        1: "Potential Active Users",
        2: "High-Value Buyers",
        3: "Churn-Risk Users",
    }
    cluster_means["user_type"] = cluster_means.index.map(cluster_names)

    cluster_means = cluster_means[
        [
            "user_type",
            "sample_count",
            "reg_days",
            "browse_duration",
            "order_count",
            "avg_order_value",
        ]
    ]

    cluster_means.to_csv(
        os.path.join(OUTPUT_DIR, "07_cluster_profile.csv"), encoding="utf-8-sig"
    )

    print("\n=== Cluster Profile ===")
    print(cluster_means.round(2))

    # 14. Cluster profile bar charts
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 2, 1)
    plt.bar(cluster_means["user_type"], cluster_means["reg_days"])
    plt.title("Average Registration Days")
    plt.xticks(rotation=15)

    plt.subplot(2, 2, 2)
    plt.bar(cluster_means["user_type"], cluster_means["browse_duration"])
    plt.title("Average Browsing Duration")
    plt.xticks(rotation=15)

    plt.subplot(2, 2, 3)
    plt.bar(cluster_means["user_type"], cluster_means["order_count"])
    plt.title("Average Order Count")
    plt.xticks(rotation=15)

    plt.subplot(2, 2, 4)
    plt.bar(cluster_means["user_type"], cluster_means["avg_order_value"])
    plt.title("Average Order Value")
    plt.xticks(rotation=15)

    plt.tight_layout()
    save_plot("08_cluster_profile_bar.png")

    print(f"\nExperiment finished. All figures have been saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
