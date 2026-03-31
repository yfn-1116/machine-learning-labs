import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles


def make_all_datasets(random_state=42):
    np.random.seed(random_state)

    # 1. 球形簇
    X_blobs, y_blobs = make_blobs(
        n_samples=500,
        centers=3,
        cluster_std=0.6,
        random_state=random_state
    )

    # 2. 月牙
    X_moons, y_moons = make_moons(
        n_samples=500,
        noise=0.08,
        random_state=random_state
    )

    # 3. 环形
    X_circles, y_circles = make_circles(
        n_samples=500,
        factor=0.5,
        noise=0.05,
        random_state=random_state
    )

    # 4. 混合 + 噪声
    X1, _ = make_blobs(
        n_samples=200,
        centers=[[-5, -5]],
        cluster_std=0.4,
        random_state=random_state
    )

    X2, _ = make_moons(
        n_samples=200,
        noise=0.07,
        random_state=random_state
    )
    X2 = X2 * 2 + [3, -2]

    X3, _ = make_blobs(
        n_samples=200,
        centers=[[0, 4]],
        cluster_std=0.5,
        random_state=random_state
    )

    X_noise = np.random.uniform(low=-7, high=7, size=(50, 2))
    X_mixed = np.vstack([X1, X2, X3, X_noise])

    return {
        "blobs": (X_blobs, y_blobs),
        "moons": (X_moons, y_moons),
        "circles": (X_circles, y_circles),
        "mixed": (X_mixed, None),
    }