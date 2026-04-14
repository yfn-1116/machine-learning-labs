from __future__ import annotations

import numpy as np


def gaussian_kernel(x: np.ndarray, x0: np.ndarray, tau: float) -> np.ndarray:
    """
    高斯核权重
    x:  (n_samples, n_features)
    x0: (n_features,)
    """
    diff = x - x0
    return np.exp(-np.sum(diff ** 2, axis=1) / (2 * tau ** 2))


def ols_fit_predict(x: np.ndarray, y: np.ndarray, x_query: np.ndarray) -> np.ndarray:
    """
    全局 OLS 拟合并预测
    x:       (n_samples, 1)
    y:       (n_samples,)
    x_query: (m_samples, 1)
    """
    X = np.hstack([np.ones((x.shape[0], 1)), x])
    Xq = np.hstack([np.ones((x_query.shape[0], 1)), x_query])

    theta = np.linalg.pinv(X.T @ X) @ X.T @ y
    return Xq @ theta


def lwlr_predict(x: np.ndarray, y: np.ndarray, x0: np.ndarray, tau: float = 0.15) -> float:
    """
    单点 LWLR 预测
    """
    w = gaussian_kernel(x, x0, tau)
    W = np.diag(w)

    X = np.hstack([np.ones((x.shape[0], 1)), x])
    X0 = np.hstack([[1.0], x0])

    theta = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ y
    return float(X0 @ theta)


def lwlr_predict_all(x: np.ndarray, y: np.ndarray, tau: float = 0.15) -> np.ndarray:
    """
    对所有样本点做 LWLR 预测
    """
    y_hat = np.zeros(len(x), dtype=float)
    for i in range(len(x)):
        y_hat[i] = lwlr_predict(x, y, x[i], tau=tau)
    return y_hat


def _tricube(u: np.ndarray) -> np.ndarray:
    """
    LOWESS 的距离权重函数
    """
    out = np.zeros_like(u, dtype=float)
    mask = np.abs(u) < 1
    out[mask] = (1 - np.abs(u[mask]) ** 3) ** 3
    return out


def _bisquare(u: np.ndarray) -> np.ndarray:
    """
    robust bisquare 权重
    """
    out = np.zeros_like(u, dtype=float)
    mask = np.abs(u) < 1
    out[mask] = (1 - u[mask] ** 2) ** 2
    return out


def lowess(
    x: np.ndarray,
    y: np.ndarray,
    frac: float = 0.3,
    it: int = 3,
) -> np.ndarray:
    """
    原生 LOWESS 平滑
    x: 一维向量
    y: 一维向量
    frac: 每次局部拟合使用的邻近样本比例
    it: robust 迭代次数
    """
    x = np.asarray(x, dtype=float).copy()
    y = np.asarray(y, dtype=float).copy()

    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    n = len(x_sorted)
    r = max(2, int(np.ceil(frac * n)))
    y_smoothed = np.zeros(n, dtype=float)
    robust_weights = np.ones(n, dtype=float)

    for _ in range(it):
        for i in range(n):
            dist = np.abs(x_sorted - x_sorted[i])
            nearest_idx = np.argsort(dist)[:r]
            max_dist = dist[nearest_idx[-1]]

            if max_dist == 0:
                distance_weights = np.ones(r, dtype=float)
            else:
                u = dist[nearest_idx] / max_dist
                distance_weights = _tricube(u)

            local_weights = distance_weights * robust_weights[nearest_idx]

            X_local = np.column_stack([np.ones(r), x_sorted[nearest_idx]])
            y_local = y_sorted[nearest_idx]
            W = np.diag(local_weights)

            try:
                beta = np.linalg.pinv(X_local.T @ W @ X_local) @ X_local.T @ W @ y_local
                y_smoothed[i] = beta[0] + beta[1] * x_sorted[i]
            except np.linalg.LinAlgError:
                y_smoothed[i] = np.mean(y_local)

        residuals = y_sorted - y_smoothed
        mad = np.median(np.abs(residuals))

        if mad < 1e-12:
            break

        u = residuals / (6.0 * mad)
        robust_weights = _bisquare(u)

    y_final = np.zeros(n, dtype=float)
    y_final[order.argsort()] = y_smoothed
    return y_final