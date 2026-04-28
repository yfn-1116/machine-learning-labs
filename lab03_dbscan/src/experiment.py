import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN as SklearnDBSCAN
from sklearn.neighbors import NearestNeighbors

from dbscan import NativeDBSCAN
from datasets import make_all_datasets


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(BASE_DIR, "figures")
OUT_DIR = os.path.join(BASE_DIR, "figures")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams["font.family"] = ["SimHei", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def save_fig(filename):
    path = os.path.join(FIG_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[Saved] {path}")


def plot_raw_datasets(datasets):
    plt.figure(figsize=(12, 10))

    for i, (name, (X, _)) in enumerate(datasets.items(), 1):
        plt.subplot(2, 2, i)
        plt.scatter(X[:, 0], X[:, 1], s=20)
        plt.title(f"Raw dataset: {name}")
        plt.grid(alpha=0.3)

    plt.tight_layout()
    save_fig("raw_datasets.png")
    plt.show()


def plot_k_distance(X, min_samples, title, filename):
    neigh = NearestNeighbors(n_neighbors=min_samples - 1)
    neigh.fit(X)
    distances, _ = neigh.kneighbors(X)

    k_distances = np.sort(distances[:, -1])[::-1]

    plt.figure(figsize=(8, 5))
    plt.plot(k_distances)
    plt.title(title)
    plt.xlabel("Sorted sample index")
    plt.ylabel("k-distance")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_fig(filename)
    plt.show()


def plot_cluster_result(X, labels, core_indices, title, filename):
    noise_mask = labels == -1
    core_mask = np.isin(np.arange(len(X)), core_indices)
    border_mask = ~noise_mask & ~core_mask

    plt.figure(figsize=(8, 6))

    plt.scatter(
        X[core_mask, 0],
        X[core_mask, 1],
        c=labels[core_mask],
        s=60,
        marker="o",
        label="core",
    )
    plt.scatter(
        X[border_mask, 0],
        X[border_mask, 1],
        c=labels[border_mask],
        s=25,
        marker="s",
        label="border",
    )
    plt.scatter(
        X[noise_mask, 0], X[noise_mask, 1], c="black", s=20, marker="x", label="noise"
    )

    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_fig(filename)
    plt.show()


def run_native_dbscan(datasets):
    params = {
        "blobs": {"eps": 0.4, "min_samples": 5},
        "moons": {"eps": 0.18, "min_samples": 5},
        "circles": {"eps": 0.12, "min_samples": 5},
        "mixed": {"eps": 0.45, "min_samples": 5},
    }

    for name, (X, _) in datasets.items():
        model = NativeDBSCAN(**params[name])
        labels = model.fit_predict(X)

        plot_cluster_result(
            X,
            labels,
            model.core_sample_indices_,
            title=f"Native DBSCAN: {name}",
            filename=f"dbscan_{name}.png",
        )

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        print(f"{name}: clusters={n_clusters}, noise={n_noise}")

    return params


def compare_with_kmeans(datasets):
    for name in ["moons", "circles"]:
        X, _ = datasets[name]

        kmeans = KMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(X)

        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=labels, s=20)
        plt.scatter(
            kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=200,
            marker="*",
        )
        plt.title(f"KMeans: {name}")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        save_fig(f"kmeans_{name}.png")
        plt.show()


def compare_with_sklearn(datasets, params):
    X, _ = datasets["mixed"]

    native = NativeDBSCAN(**params["mixed"])
    native_labels = native.fit_predict(X)

    sklearn_model = SklearnDBSCAN(**params["mixed"])
    sklearn_labels = sklearn_model.fit_predict(X)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=native_labels, s=20)
    plt.title("Native DBSCAN")

    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=sklearn_labels, s=20)
    plt.title("sklearn DBSCAN")

    plt.tight_layout()
    save_fig("compare_native_vs_sklearn.png")
    plt.show()


def main():
    datasets = make_all_datasets()

    # 1. 原始数据图
    plot_raw_datasets(datasets)

    # 2. k-distance 图
    for name, (X, _) in datasets.items():
        plot_k_distance(
            X,
            min_samples=5,
            title=f"k-distance: {name}",
            filename=f"k_distance_{name}.png",
        )

    # 3. 原生 DBSCAN
    params = run_native_dbscan(datasets)

    # 4. KMeans 对比
    compare_with_kmeans(datasets)

    # 5. sklearn 对比
    compare_with_sklearn(datasets, params)


if __name__ == "__main__":
    main()
