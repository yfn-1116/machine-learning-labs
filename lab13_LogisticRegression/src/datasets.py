from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "user_churn_data.csv"
FIG_DIR = PROJECT_ROOT / "figures" / "figures"

BASE_FEATURE_NAMES = [
    "注册时长", "最近活跃天数", "活跃频率", "访问时长", "使用功能数",
    "累计消费金额", "最近消费金额", "消费频率", "客单价",
    "评论数", "分享数", "点赞数", "反馈次数", "客服咨询次数",
    "投诉次数", "评分均值",
]

CLASS_NAMES = ["留存", "流失"]


class StandardScalerNative:
    """
    原生Z-score标准化。
    注意：只在训练集fit，再transform测试集，避免数据泄露。
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def ensure_dirs():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(data_path=DATA_PATH):
    ensure_dirs()
    data_path = Path(data_path)
    if not data_path.exists():
        from generate_dataset import generate_dataset
        generate_dataset(data_path)
    return pd.read_csv(data_path)


def winsorize_iqr(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return series.clip(low, high)


def preprocess_dataframe(df):
    """
    数据预处理：
    1. 缺失值中位数填充
    2. IQR轻度截断异常值
    3. 金额类和计数类特征做log1p
    4. 构造少量业务特征
    """
    df = df.copy()

    if "是否流失" not in df.columns:
        raise ValueError("数据集必须包含目标列：是否流失")

    feature_cols = [c for c in BASE_FEATURE_NAMES if c in df.columns]
    missing = sorted(set(BASE_FEATURE_NAMES) - set(feature_cols))
    if missing:
        raise ValueError(f"数据集缺少必要特征列: {missing}")

    X_df = df[feature_cols].copy()

    for col in X_df.columns:
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce")
        X_df[col] = X_df[col].fillna(X_df[col].median())

    for col in X_df.columns:
        X_df[col] = winsorize_iqr(X_df[col])

    log_cols = [
        "累计消费金额", "最近消费金额", "客单价",
        "评论数", "分享数", "点赞数",
    ]
    for col in log_cols:
        if col in X_df.columns:
            X_df[col] = np.log1p(np.clip(X_df[col], 0, None))

    X_df["消费活跃比"] = X_df["消费频率"] / (X_df["活跃频率"].abs() + 1e-6)
    X_df["互动总量"] = X_df["评论数"] + X_df["分享数"] + X_df["点赞数"]
    X_df["投诉咨询比"] = X_df["投诉次数"] / (X_df["客服咨询次数"].abs() + 1.0)

    y = df["是否流失"].astype(int).to_numpy()
    user_ids = df["用户ID"].to_numpy() if "用户ID" in df.columns else np.arange(len(df))

    return X_df, y, user_ids


def prepare_train_test(df, test_size=0.3, random_state=42):
    X_df, y, user_ids = preprocess_dataframe(df)

    X_train_df, X_test_df, y_train, y_test, uid_train, uid_test = train_test_split(
        X_df,
        y,
        user_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScalerNative()
    X_train = scaler.fit_transform(X_train_df.to_numpy(dtype=float))
    X_test = scaler.transform(X_test_df.to_numpy(dtype=float))

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "uid_train": uid_train,
        "uid_test": uid_test,
        "feature_names": X_train_df.columns.tolist(),
        "scaler": scaler,
        "X_train_df": X_train_df,
        "X_test_df": X_test_df,
    }
