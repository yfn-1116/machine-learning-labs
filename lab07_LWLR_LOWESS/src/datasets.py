from __future__ import annotations

import numpy as np
import pandas as pd


def generate_abtest_data(random_seed: int = 42) -> pd.DataFrame:
    """
    生成 28 天 AB 实验模拟数据
    字段:
        day: 天数
        control: 对照组付费率
        experiment: 实验组付费率
        is_weekend: 是否周末
    """
    rng = np.random.default_rng(random_seed)

    days = np.arange(1, 29)

    # 简单定义周末/周期峰值日
    weekend_days = {7, 8, 14, 15, 21, 22, 28}
    is_weekend = np.array([1 if d in weekend_days else 0 for d in days], dtype=int)

    # -----------------------------
    # 对照组：整体稳定 + 周末波动 + 噪声 + 异常点
    # -----------------------------
    control_base = 0.0105 + rng.normal(0, 0.00018, size=len(days))
    control_weekend_effect = is_weekend * 0.0038
    control = control_base + control_weekend_effect

    # 异常点：第10天骤降，第18天误报升高
    control[9] = 0.0070   # day=10
    control[17] = 0.0160  # day=18

    # -----------------------------
    # 实验组：第3天开始生效，第7天后进一步增强
    # -----------------------------
    exp = np.zeros_like(control)

    for i, d in enumerate(days):
        if d <= 2:
            trend = 0.0102 + 0.00012 * d
        elif 3 <= d <= 7:
            trend = 0.0108 + 0.00050 * (d - 3)
        elif 8 <= d <= 14:
            trend = 0.0130 + 0.00033 * (d - 8)
        else:
            trend = 0.0151 + 0.00003 * (d - 15)

        exp[i] = trend

    exp += is_weekend * 0.0018
    exp += rng.normal(0, 0.00015, size=len(days))

    df = pd.DataFrame(
        {
            "day": days,
            "control": control,
            "experiment": exp,
            "is_weekend": is_weekend,
        }
    )
    return df


def standardize_days(days: np.ndarray) -> np.ndarray:
    """
    将 day 标准化到 [0, 1]
    """
    days = np.asarray(days, dtype=float)
    return (days - days.min()) / (days.max() - days.min())