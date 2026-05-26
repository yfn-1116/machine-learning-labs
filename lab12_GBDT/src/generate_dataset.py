from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "live_stream_conversion_data.csv"

FEATURE_NAMES = [
    "场次在线峰值", "场均观看人数", "平均观看时长", "点赞数", "评论数", "分享数", "商品点击数",
    "讲解商品数量", "场均GMV", "客单价", "退货率", "粉丝占比", "付费流量占比",
]

CLASS_NAMES = ["低转化", "普通转化", "高转化", "爆款转化"]


def rate_to_label(rate: float) -> int:
    if rate < 2:
        return 0
    if rate < 5:
        return 1
    if rate < 10:
        return 2
    return 3


def generate_dataset(output_path=DATA_PATH, n_samples=360, random_state=42):
    """
    生成电商直播流量转化率预测模拟数据。
    如果你后面有真实数据，直接替换 data/live_stream_conversion_data.csv 即可。
    """
    rng = np.random.default_rng(random_state)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    categories = np.array(["服饰", "美妆", "食品", "数码"])
    time_slots = np.array(["早场", "午场", "晚场", "深夜场"])
    styles = np.array(["测评型", "福利型", "专业讲解型", "娱乐互动型"])

    levels = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.28, 0.34, 0.25, 0.13])
    rows = []

    for level in levels:
        category = rng.choice(categories)
        time_slot = rng.choice(time_slots, p=[0.18, 0.22, 0.45, 0.15])
        style = rng.choice(styles)

        base_peak = [5500, 11000, 21000, 42000][level]
        base_watchers = [26000, 58000, 115000, 220000][level]
        base_gmv = [65000, 180000, 480000, 980000][level]

        slot_factor = 1.18 if time_slot == "晚场" else 0.86 if time_slot == "深夜场" else 1.0
        interact_factor = 1.15 if style in ["福利型", "娱乐互动型"] else 1.0

        online_peak = int(max(1200, rng.lognormal(np.log(base_peak * slot_factor), 0.28)))
        avg_viewers = int(max(5000, rng.lognormal(np.log(base_watchers * slot_factor), 0.25)))
        avg_watch_time = float(np.clip(rng.normal([16, 23, 32, 43][level], 4.0), 8, 60))
        likes = int(max(3000, rng.lognormal(np.log([35000, 85000, 180000, 360000][level] * interact_factor), 0.35)))
        comments = int(max(300, rng.lognormal(np.log([800, 1800, 4300, 9000][level] * interact_factor), 0.35)))
        shares = int(max(100, rng.lognormal(np.log([180, 480, 1100, 2600][level] * interact_factor), 0.38)))
        clicks = int(max(800, rng.lognormal(np.log([4200, 10500, 26000, 60000][level]), 0.32)))
        product_count = int(np.clip(rng.normal([6, 9, 13, 18][level], 2.0), 3, 28))
        gmv = float(max(12000, rng.lognormal(np.log(base_gmv * slot_factor), 0.35)))
        unit_price = float(np.clip(rng.normal([55, 76, 105, 145][level], 18), 19, 399))
        return_rate = float(np.clip(rng.normal([0.27, 0.22, 0.17, 0.12][level], 0.035), 0.03, 0.42))
        fan_ratio = float(np.clip(rng.normal([0.76, 0.66, 0.55, 0.45][level], 0.08), 0.15, 0.95))
        paid_ratio = float(np.clip(rng.normal([0.18, 0.28, 0.41, 0.55][level], 0.08), 0.03, 0.90))

        if level == 0:
            conversion_rate = rng.uniform(0.5, 1.9)
        elif level == 1:
            conversion_rate = rng.uniform(2.1, 4.8)
        elif level == 2:
            conversion_rate = rng.uniform(5.2, 9.6)
        else:
            conversion_rate = rng.uniform(10.3, 16.5)

        conversion_rate += 0.35 * (time_slot == "晚场") + 0.25 * (style == "福利型") - 0.45 * (return_rate > 0.25)
        conversion_rate = float(np.clip(conversion_rate, 0.3, 18.0))
        label = rate_to_label(conversion_rate)

        rows.append({
            "品类": category,
            "时段": time_slot,
            "主播风格": style,
            "场次在线峰值": online_peak,
            "场均观看人数": avg_viewers,
            "平均观看时长": round(avg_watch_time, 2),
            "点赞数": likes,
            "评论数": comments,
            "分享数": shares,
            "商品点击数": clicks,
            "讲解商品数量": product_count,
            "场均GMV": round(gmv, 2),
            "客单价": round(unit_price, 2),
            "退货率": round(return_rate, 4),
            "粉丝占比": round(fan_ratio, 4),
            "付费流量占比": round(paid_ratio, 4),
            "转化率": round(conversion_rate, 3),
            "标签": label,
        })

    df = pd.DataFrame(rows)

    missing_cols = ["场次在线峰值", "场均观看人数", "评论数", "场均GMV", "退货率", "粉丝占比"]
    for col in missing_cols:
        mask = rng.random(n_samples) < 0.015
        df.loc[mask, col] = np.nan

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"数据集已生成: {output_path}")
    print(f"样本数: {len(df)}，标签分布: {df['标签'].value_counts().sort_index().to_dict()}")
    return df


if __name__ == "__main__":
    generate_dataset()
