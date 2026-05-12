# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_NAMES = [
    "近30天访客数",
    "点击率",
    "30天销量",
    "动销率",
    "毛利率",
    "客单价",
    "库存周转天数",
    "现货库存",
    "投放ROI",
]

TARGET_NAME = "标签"

CLASS_NAMES = {
    0: "引流款",
    1: "利润款",
    2: "形象款",
    3: "滞销清库存款",
}

CONTINUOUS_FEATURES = list(FEATURE_NAMES)

CATEGORICAL_FEATURES = []


def create_demo_dataset(save_path, n_samples=200, random_state=42):
    rng = np.random.default_rng(random_state)
    rows = []

    for _ in range(n_samples):
        label = rng.choice([0, 1, 2, 3], p=[0.30, 0.30, 0.15, 0.25])

        if label == 0:
            visitors = rng.integers(800, 3000)
            click_rate = rng.uniform(0.06, 0.15)
            sales = rng.integers(150, 500)
            sell_rate = rng.uniform(0.7, 1.0)
            margin = rng.uniform(0.08, 0.20)
            price = rng.uniform(29.9, 79.9)
            turnover_days = rng.integers(10, 25)
            stock = rng.integers(80, 250)
            roi = rng.uniform(1.5, 3.5)
        elif label == 1:
            visitors = rng.integers(200, 600)
            click_rate = rng.uniform(0.04, 0.08)
            sales = rng.integers(50, 150)
            sell_rate = rng.uniform(0.5, 0.8)
            margin = rng.uniform(0.35, 0.55)
            price = rng.uniform(99.9, 199.9)
            turnover_days = rng.integers(15, 35)
            stock = rng.integers(50, 150)
            roi = rng.uniform(2.5, 5.0)
        elif label == 2:
            visitors = rng.integers(50, 200)
            click_rate = rng.uniform(0.02, 0.05)
            sales = rng.integers(5, 30)
            sell_rate = rng.uniform(0.1, 0.35)
            margin = rng.uniform(0.55, 0.80)
            price = rng.uniform(249.9, 499.9)
            turnover_days = rng.integers(35, 60)
            stock = rng.integers(10, 50)
            roi = rng.uniform(0.5, 1.5)
        else:
            visitors = rng.integers(10, 80)
            click_rate = rng.uniform(0.005, 0.025)
            sales = rng.integers(0, 10)
            sell_rate = rng.uniform(0.02, 0.15)
            margin = rng.uniform(0.05, 0.30)
            price = rng.uniform(19.9, 69.9)
            turnover_days = rng.integers(50, 90)
            stock = rng.integers(150, 500)
            roi = rng.uniform(0.1, 0.6)

        rows.append({
            "近30天访客数": int(max(0, visitors)),
            "点击率": round(float(max(0.001, click_rate)), 4),
            "30天销量": int(max(0, sales)),
            "动销率": round(float(max(0.0, min(1.0, sell_rate))), 4),
            "毛利率": round(float(max(0.0, min(1.0, margin))), 4),
            "客单价": round(float(max(9.9, price)), 2),
            "库存周转天数": int(max(1, turnover_days)),
            "现货库存": int(max(0, stock)),
            "投放ROI": round(float(max(0.0, roi)), 2),
            "标签": int(label),
        })

    df = pd.DataFrame(rows)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    return df


def read_csv_safely(data_path):
    for enc in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            return pd.read_csv(data_path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError("CSV 读取失败，请检查编码。")


def load_dataset(data_path):
    data_path = Path(data_path)
    if not data_path.exists():
        print("[提示] 未找到数据文件，自动生成模拟数据。")
        return create_demo_dataset(data_path)
    return read_csv_safely(data_path)


def preprocess_dataset(df):
    df = df.copy()
    df = df[FEATURE_NAMES + [TARGET_NAME]]
    df = df.dropna(subset=[TARGET_NAME])
    df[TARGET_NAME] = df[TARGET_NAME].astype(int)

    for col in CONTINUOUS_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].mean())

    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]
    return X, y


def split_train_test(X, y, test_size=0.3, random_state=42):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def show_basic_info(df):
    print("\n========== 数据集基本信息 ==========")
    print("样本数量：", len(df))
    print("字段：", list(df.columns))

    print("\n========== 缺失值统计 ==========")
    print(df.isnull().sum())

    print("\n========== 标签分布 ==========")
    print(df[TARGET_NAME].value_counts().sort_index())
