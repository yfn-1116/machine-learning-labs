# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_NAMES = [
    "最近一次消费天数",
    "月消费频率",
    "平均单次消费金额",
    "单次平均到店时长",
    "是否参与团课",
    "会员等级",
]

TARGET_NAME = "标签"

CLASS_NAMES = {
    0: "高复购用户",
    1: "低活跃用户",
    2: "流失风险用户",
}

CONTINUOUS_FEATURES = [
    "最近一次消费天数",
    "月消费频率",
    "平均单次消费金额",
    "单次平均到店时长",
]

CATEGORICAL_FEATURES = [
    "是否参与团课",
    "会员等级",
]


def create_demo_dataset(save_path, n_samples=300, random_state=42):
    rng = np.random.default_rng(random_state)
    rows = []

    for _ in range(n_samples):
        label = rng.choice([0, 1, 2], p=[0.35, 0.30, 0.35])

        if label == 0:
            r = rng.integers(1, 15)
            f = rng.integers(5, 13)
            m = rng.normal(160, 25)
            duration = rng.normal(70, 12)
            group = rng.choice(["是", "否"], p=[0.75, 0.25])
            level = rng.choice(["白银", "黄金", "钻石"], p=[0.2, 0.5, 0.3])
        elif label == 1:
            r = rng.integers(15, 40)
            f = rng.integers(1, 5)
            m = rng.normal(110, 20)
            duration = rng.normal(45, 10)
            group = rng.choice(["是", "否"], p=[0.35, 0.65])
            level = rng.choice(["普通", "白银", "黄金"], p=[0.55, 0.35, 0.1])
        else:
            r = rng.integers(35, 90)
            f = rng.integers(0, 3)
            m = rng.normal(90, 18)
            duration = rng.normal(30, 8)
            group = rng.choice(["是", "否"], p=[0.15, 0.85])
            level = rng.choice(["普通", "白银"], p=[0.75, 0.25])

        rows.append({
            "最近一次消费天数": int(max(0, r)),
            "月消费频率": int(max(0, f)),
            "平均单次消费金额": round(float(max(20, m)), 2),
            "单次平均到店时长": round(float(max(10, duration)), 2),
            "是否参与团课": group,
            "会员等级": level,
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
        print("[提示] 未找到 data/gym_user_behavior.csv，自动生成模拟数据。")
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

    encoders = {}

    for col in CATEGORICAL_FEATURES:
        mode_value = df[col].mode(dropna=True)
        fill_value = mode_value.iloc[0] if len(mode_value) > 0 else "未知"
        df[col] = df[col].fillna(fill_value).astype(str)

        if col == "是否参与团课":
            mapping = {"否": 0, "是": 1}
        elif col == "会员等级":
            mapping = {"普通": 0, "白银": 1, "黄金": 2, "钻石": 3}
        else:
            values = sorted(df[col].unique())
            mapping = {v: i for i, v in enumerate(values)}

        for v in df[col].unique():
            if v not in mapping:
                mapping[v] = len(mapping)

        df[col] = df[col].map(mapping).astype(int)
        encoders[col] = mapping

    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]
    return X, y, encoders


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
