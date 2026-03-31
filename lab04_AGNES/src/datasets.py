import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


def generate_user_data(random_state=42, n_samples=400):
    """
    生成4类新用户行为模拟数据
    特征：
    - reg_days: 注册天数
    - browse_duration: 浏览时长
    - order_count: 下单次数
    - avg_order_value: 平均客单价
    """
    centers = [
        [5, 5, 0, 0],       # 新客观望型
        [15, 25, 2, 50],    # 潜力活跃型
        [12, 15, 8, 200],   # 高价值下单型
        [25, 3, 0, 0],      # 流失风险型
    ]

    X, y_true = make_blobs(
        n_samples=n_samples,
        n_features=4,
        centers=centers,
        cluster_std=[1.5, 2.0, 1.8, 1.2],
        random_state=random_state,
    )

    df = pd.DataFrame(
        X,
        columns=["reg_days", "browse_duration", "order_count", "avg_order_value"]
    )
    df["true_label"] = y_true

    # 防止出现负值
    df["order_count"] = df["order_count"].clip(lower=0)
    df["avg_order_value"] = df["avg_order_value"].clip(lower=0)

    return df


def remove_outliers(df, features, q=0.99):
    """
    按分位数法过滤异常值
    """
    df_clean = df.copy()
    for feature in features:
        upper = df_clean[feature].quantile(q)
        lower = df_clean[feature].quantile(1 - q)
        df_clean = df_clean[
            (df_clean[feature] >= lower) & (df_clean[feature] <= upper)
        ]
    return df_clean.reset_index(drop=True)


def standardize_features(df, feature_cols):
    """
    标准化特征
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    return X_scaled, scaler