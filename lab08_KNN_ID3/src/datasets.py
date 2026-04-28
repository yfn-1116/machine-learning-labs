from pathlib import Path
import shutil

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DEFAULT_CSV_PATH = DATA_DIR / "live_ecommerce_user_intention.csv"
LEGACY_DATA_PATH = BASE_DIR / "src" / "data" / "live_ecommerce_user_intention.csv"

# KNN 使用的原始数值特征
FEATURES = ["停留时长", "小黄车点击次数", "互动次数"]
LABEL = "标签"


def generate_live_ecommerce_data(n_samples=1000, random_state=42):
    """
    生成直播电商用户行为数据。

    特征：
    1. 停留时长
    2. 小黄车点击次数
    3. 互动次数

    标签：
    1 表示高意向下单用户
    0 表示非高意向用户
    """
    rng = np.random.default_rng(random_state)

    stay_time = rng.gamma(shape=2.2, scale=32, size=n_samples)
    stay_time = np.clip(stay_time, 1, 300)

    cart_clicks = rng.poisson(lam=np.clip(stay_time / 45, 0.2, 8))
    cart_clicks = np.clip(cart_clicks, 0, 20)

    interactions = rng.poisson(lam=np.clip(stay_time / 30, 0.3, 12))
    interactions = np.clip(interactions, 0, 30)

    score = (
        0.035 * stay_time
        + 0.9 * cart_clicks
        + 0.45 * interactions
        + rng.normal(0, 2.2, n_samples)
    )
    label = (score >= 7.5).astype(int)

    return pd.DataFrame(
        {
            "停留时长": stay_time.round(2),
            "小黄车点击次数": cart_clicks.astype(int),
            "互动次数": interactions.astype(int),
            "标签": label,
        }
    )


def load_or_create_data(csv_path=DEFAULT_CSV_PATH):
    """
    读取数据集；若不存在，则自动生成一份模拟数据。
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists() and LEGACY_DATA_PATH.exists():
        shutil.move(str(LEGACY_DATA_PATH), str(csv_path))
    elif csv_path.exists() and LEGACY_DATA_PATH.exists():
        LEGACY_DATA_PATH.unlink()
        try:
            LEGACY_DATA_PATH.parent.rmdir()
        except OSError:
            pass

    if csv_path.exists():
        return pd.read_csv(csv_path)

    data = generate_live_ecommerce_data()
    data.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return data


def preprocess_for_knn(data):
    """
    KNN 数据预处理。

    KNN 可以直接处理连续型特征，因此保留原始数值特征。
    主要处理：
    1. 缺失值
    2. 异常值
    """
    processed = data.copy()

    processed["停留时长"] = processed["停留时长"].fillna(processed["停留时长"].mean())
    processed["小黄车点击次数"] = processed["小黄车点击次数"].fillna(
        processed["小黄车点击次数"].mode()[0]
    )
    processed["互动次数"] = processed["互动次数"].fillna(
        processed["互动次数"].mode()[0]
    )

    processed["停留时长"] = processed["停留时长"].clip(0, 300)
    processed["小黄车点击次数"] = processed["小黄车点击次数"].clip(0, 20)
    processed["互动次数"] = processed["互动次数"].clip(0, 30)

    return processed


def bucket_stay_time(value):
    """
    对停留时长进行分桶，供 ID3 使用。
    """
    if value < 30:
        return "短停留"
    if value < 60:
        return "中停留"
    return "长停留"


def bucket_count(value):
    """
    对点击次数、互动次数进行分桶，减少树分支数量。
    """
    if value == 0:
        return "无"
    if value <= 3:
        return "低"
    if value <= 7:
        return "中"
    return "高"


def preprocess_for_id3(data):
    """
    ID3 数据预处理。

    ID3 需要离散特征，因此把原始连续特征映射为分桶类别。
    """
    knn_ready = preprocess_for_knn(data)

    return pd.DataFrame(
        {
            "停留时长分桶": knn_ready["停留时长"].apply(bucket_stay_time),
            "小黄车点击次数分桶": knn_ready["小黄车点击次数"].apply(bucket_count),
            "互动次数分桶": knn_ready["互动次数"].apply(bucket_count),
            "标签": knn_ready["标签"],
        }
    )
