from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "live_stream_conversion_data.csv"
FIG_DIR = PROJECT_ROOT / "figures" / "figures"

FEATURE_NAMES = [
    "场次在线峰值", "场均观看人数", "平均观看时长", "点赞数", "评论数", "分享数", "商品点击数",
    "讲解商品数量", "场均GMV", "客单价", "退货率", "粉丝占比", "付费流量占比",
]
CLASS_NAMES = ["低转化", "普通转化", "高转化", "爆款转化"]
RATIO_FEATURES = ["退货率", "粉丝占比", "付费流量占比"]


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


def label_from_rate(rate):
    if rate < 2:
        return 0
    if rate < 5:
        return 1
    if rate < 10:
        return 2
    return 3


def preprocess_dataset(df):
    """
    缺失值填充 + 轻度异常值截断。
    注意：不删除爆款直播样本，因为这些极端样本正是实验要重点识别的对象。
    """
    df = df.copy()
    if "标签" not in df.columns and "转化率" in df.columns:
        df["标签"] = df["转化率"].apply(label_from_rate)

    selected_features = [c for c in FEATURE_NAMES if c in df.columns]
    if len(selected_features) != len(FEATURE_NAMES):
        missing = sorted(set(FEATURE_NAMES) - set(selected_features))
        raise ValueError(f"数据集缺少必要特征列: {missing}")

    X_df = df[selected_features].copy()

    missing_ratio = X_df.isnull().mean()
    keep_cols = [c for c in selected_features if missing_ratio[c] <= 0.30]
    X_df = X_df[keep_cols]

    for col in X_df.columns:
        if col in RATIO_FEATURES:
            X_df[col] = X_df[col].fillna(X_df[col].mean())
        else:
            X_df[col] = X_df[col].fillna(X_df[col].median())

    for col in X_df.columns:
        q01, q99 = X_df[col].quantile([0.01, 0.99])
        X_df[col] = X_df[col].clip(q01, q99)

    y = df["标签"].astype(int).to_numpy()
    X = X_df.to_numpy(dtype=float)
    return X, y, list(X_df.columns), df


def get_train_test(X, y, test_size=0.30, random_state=42):
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
