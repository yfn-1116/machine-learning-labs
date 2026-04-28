# -*- coding: utf-8 -*-
"""
datasets.py
生成“新品销量预测”模拟数据集，并支持保存为 CSV。
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd


RANDOM_SEED = 42

FEATURE_NAMES: List[str] = [
    "广告投放费用",
    "线上曝光量",
    "线下地推费用",
    "铺货门店数",
    "渠道覆盖率",
    "货架占比",
    "本品定价",
    "竞品均价",
    "促销折扣率",
    "赠品数量",
    "品牌知名度",
    "产品好评率",
    "节假日影响",
    "区域消费水平",
    "社群触达人数",
]

TARGET_NAME = "新品月销量"


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_sales_dataset(
    n_samples: int = 50,
    random_state: int = RANDOM_SEED,
    save_csv: bool = True,
    csv_path: str | Path | None = None,
    add_missing_ratio: float = 0.03,
    add_outlier_ratio: float = 0.02,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    生成新品销量预测模拟数据集。

    返回:
        df: 包含特征与标签的数据集
        true_beta: 真实系数（用于实验对比）
    """
    rng = np.random.default_rng(random_state)

    # 1. 强相关营销类特征
    ad_cost = rng.normal(10, 3, n_samples)  # 广告投放费用（万元）
    exposure = ad_cost * 50 + rng.normal(0, 20, n_samples)  # 与广告投放强相关
    promotion_cost = ad_cost * 0.3 + rng.normal(0, 0.5, n_samples)  # 与广告投放强相关

    # 2. 强相关渠道类特征
    store_num = rng.integers(10, 100, n_samples)  # 铺货门店数
    channel_coverage = store_num / 100 + rng.normal(0, 0.05, n_samples)
    shelf_ratio = channel_coverage * 0.5 + rng.normal(0, 0.03, n_samples)

    # 3. 相关定价类特征
    price = rng.normal(50, 10, n_samples)  # 本品定价
    competitor_price = price * 0.9 + rng.normal(0, 3, n_samples)  # 竞品均价

    # 4. 其他独立特征
    discount = rng.uniform(0.7, 1.0, n_samples)  # 促销折扣率（1=不打折）
    gift_num = rng.integers(0, 5, n_samples)  # 赠品数量
    brand_score = rng.normal(50, 10, n_samples)  # 品牌知名度
    good_rate = rng.uniform(0.8, 0.99, n_samples)  # 产品好评率
    holiday = rng.integers(0, 2, n_samples)  # 是否节假日
    consumption_level = rng.normal(50, 15, n_samples)  # 区域消费水平
    community_reach = rng.normal(10, 3, n_samples)  # 社群触达人数（万人）

    X = np.column_stack(
        [
            ad_cost,
            exposure,
            promotion_cost,
            store_num,
            channel_coverage,
            shelf_ratio,
            price,
            competitor_price,
            discount,
            gift_num,
            brand_score,
            good_rate,
            holiday,
            consumption_level,
            community_reach,
        ]
    )

    # 真实系数：符合业务逻辑
    true_beta = np.array(
        [50, 0.2, 30, 2, 100, 80, -3, 2, 200, 15, 3, 10, 50, 2, 4], dtype=float
    )

    noise = rng.normal(0, 100, n_samples)
    y = X @ true_beta + noise

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df[TARGET_NAME] = y

    # 注入少量缺失值，便于演示缺失值处理
    if add_missing_ratio > 0:
        n_missing = max(1, int(df.shape[0] * (df.shape[1] - 1) * add_missing_ratio))
        for _ in range(n_missing):
            r = rng.integers(0, df.shape[0])
            c = rng.integers(0, len(FEATURE_NAMES))
            df.iat[r, c] = np.nan

    # 注入少量异常值，便于演示 IQR 裁剪
    if add_outlier_ratio > 0:
        n_outliers = max(1, int(df.shape[0] * add_outlier_ratio))
        cols_for_outlier = [
            "广告投放费用",
            "线上曝光量",
            "铺货门店数",
            "本品定价",
            "区域消费水平",
        ]
        for _ in range(n_outliers):
            r = rng.integers(0, df.shape[0])
            col = cols_for_outlier[rng.integers(0, len(cols_for_outlier))]
            if pd.notna(df.loc[r, col]):
                df.loc[r, col] = df.loc[r, col] * rng.uniform(1.8, 2.5)

    if save_csv:
        if csv_path is None:
            project_root = Path(__file__).resolve().parents[1]
            csv_path = project_root / "figures" / "新品销量预测数据集.csv"
        csv_path = Path(csv_path)
        ensure_dir(csv_path.parent)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[OK] 已保存数据集: {csv_path}")

    return df, true_beta


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    print(f"[OK] 已读取数据集: {csv_path}")
    return df


if __name__ == "__main__":
    generate_sales_dataset()
