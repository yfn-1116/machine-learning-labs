# -*- coding: utf-8 -*-
"""
ridge.py
原生实现 OLS 闭式解、Ridge 闭式解、预测、评估指标、岭迹图系数轨迹。
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np


def _to_numpy_1d(y) -> np.ndarray:
    arr = np.asarray(y, dtype=float).reshape(-1)
    return arr


def _to_numpy_2d(X) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError("X 必须是二维矩阵")
    return arr


def ols_fit(X, y, add_intercept: bool = False, singular_tol: float = 1e12) -> Tuple[np.ndarray, float]:
    """
    原生 OLS 闭式解。
    返回:
        coef_: 系数
        intercept_: 截距
    """
    X = _to_numpy_2d(X)
    y = _to_numpy_1d(y)

    if add_intercept:
        X_design = np.column_stack([np.ones(X.shape[0]), X])
    else:
        X_design = X

    xtx = X_design.T @ X_design
    cond_number = np.linalg.cond(xtx)

    if not np.isfinite(cond_number) or cond_number > singular_tol:
        raise np.linalg.LinAlgError(
            f"OLS 矩阵接近奇异或病态，条件数过大: {cond_number:.4e}"
        )

    try:
        beta = np.linalg.solve(xtx, X_design.T @ y)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(f"OLS 求解失败: {exc}") from exc

    if add_intercept:
        intercept_ = float(beta[0])
        coef_ = beta[1:]
    else:
        intercept_ = 0.0
        coef_ = beta

    return coef_, intercept_


def ridge_fit(X, y, lam: float, add_intercept: bool = False) -> Tuple[np.ndarray, float]:
    """
    原生 Ridge 闭式解。
    """
    if lam < 0:
        raise ValueError("正则化参数 lam 必须 >= 0")

    X = _to_numpy_2d(X)
    y = _to_numpy_1d(y)

    if add_intercept:
        X_design = np.column_stack([np.ones(X.shape[0]), X])
        n_features = X_design.shape[1]
        I = np.eye(n_features)
        I[0, 0] = 0.0
    else:
        X_design = X
        n_features = X_design.shape[1]
        I = np.eye(n_features)

    xtx = X_design.T @ X_design
    ridge_matrix = xtx + lam * I

    try:
        beta = np.linalg.solve(ridge_matrix, X_design.T @ y)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(f"Ridge 求解失败: {exc}") from exc

    if add_intercept:
        intercept_ = float(beta[0])
        coef_ = beta[1:]
    else:
        intercept_ = 0.0
        coef_ = beta

    return coef_, intercept_


def predict(X, coef_, intercept_: float = 0.0) -> np.ndarray:
    X = _to_numpy_2d(X)
    coef_ = _to_numpy_1d(coef_)
    return X @ coef_ + float(intercept_)


def mse(y_true, y_pred) -> float:
    y_true = _to_numpy_1d(y_true)
    y_pred = _to_numpy_1d(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred) -> float:
    y_true = _to_numpy_1d(y_true)
    y_pred = _to_numpy_1d(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score_native(y_true, y_pred) -> float:
    y_true = _to_numpy_1d(y_true)
    y_pred = _to_numpy_1d(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if np.isclose(ss_tot, 0):
        return 0.0
    return float(1 - ss_res / ss_tot)


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "MSE": mse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "R2": r2_score_native(y_true, y_pred),
    }


def ridge_trace(X, y, lambdas: Iterable[float], add_intercept: bool = False) -> np.ndarray:
    """
    计算岭迹图系数轨迹。
    返回 shape = (len(lambdas), n_features)
    """
    X = _to_numpy_2d(X)
    y = _to_numpy_1d(y)

    coefs = []
    for lam in lambdas:
        coef_, _ = ridge_fit(X, y, lam=lam, add_intercept=add_intercept)
        coefs.append(coef_)
    return np.asarray(coefs, dtype=float)