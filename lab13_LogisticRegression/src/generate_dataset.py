from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "user_churn_data.csv"

FEATURE_NAMES = [
    "注册时长", "最近活跃天数", "活跃频率", "访问时长", "使用功能数",
    "累计消费金额", "最近消费金额", "消费频率", "客单价",
    "评论数", "分享数", "点赞数", "反馈次数", "客服咨询次数",
    "投诉次数", "评分均值",
]


def generate_dataset(output_path=DATA_PATH, n_samples=800, random_state=42):
    """
    生成数字运营用户流失预测模拟数据。
    目标字段：是否流失，0=留存，1=流失。
    """
    rng = np.random.default_rng(random_state)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    user_id = np.arange(1, n_samples + 1)

    register_days = rng.integers(7, 1000, size=n_samples)
    recency = np.clip(rng.gamma(shape=2.0, scale=7.5, size=n_samples), 0, 60)
    frequency = np.clip(rng.normal(5.8, 2.6, size=n_samples), 0, 16)
    visit_duration = np.clip(rng.normal(13.5, 6.0, size=n_samples), 1, 45)
    feature_count = np.clip(rng.normal(5.2, 2.0, size=n_samples), 1, 12)

    total_spend = np.clip(rng.lognormal(mean=6.0, sigma=1.0, size=n_samples) - 250, 0, None)
    recent_spend = np.clip(total_spend * rng.uniform(0.0, 0.22, size=n_samples), 0, None)
    purchase_frequency = np.clip(rng.normal(1.6, 1.1, size=n_samples), 0, 8)
    avg_order_value = np.where(
        purchase_frequency > 0.2,
        np.clip(total_spend / (purchase_frequency * rng.uniform(3, 10, size=n_samples)), 0, 800),
        0
    )

    comments = rng.poisson(np.clip(2 + frequency * 0.9, 0.5, 20))
    shares = rng.poisson(np.clip(1 + frequency * 0.45, 0.3, 12))
    likes = rng.poisson(np.clip(5 + frequency * 4.5, 1, 120))
    feedbacks = rng.poisson(0.7 + recency / 60)
    service_contacts = rng.poisson(0.5 + feedbacks * 0.55)
    complaints = rng.poisson(np.clip(0.08 + service_contacts * 0.18 + rng.random(n_samples) * 0.12, 0, 4))
    rating = np.clip(rng.normal(4.25, 0.55, size=n_samples) - complaints * 0.35, 1.0, 5.0)

    # 构建符合业务逻辑的流失风险分数：
    # 最近活跃天数越大、投诉越多、评分越低、活跃越低，流失风险越高。
    score = (
        0.080 * recency
        - 0.170 * frequency
        - 0.055 * visit_duration
        - 0.150 * feature_count
        - 0.00035 * np.log1p(total_spend) * 100
        - 0.090 * purchase_frequency
        + 0.360 * feedbacks
        + 0.520 * service_contacts
        + 1.150 * complaints
        - 0.950 * rating
        + rng.normal(0, 0.8, size=n_samples)
    )

    # 控制流失比例在25%左右，符合指导书建议的10%-30%范围。
    threshold = np.quantile(score, 0.75)
    churn = (score >= threshold).astype(int)

    df = pd.DataFrame({
        "用户ID": user_id,
        "注册时长": register_days,
        "最近活跃天数": np.round(recency, 2),
        "活跃频率": np.round(frequency, 2),
        "访问时长": np.round(visit_duration, 2),
        "使用功能数": np.round(feature_count, 2),
        "累计消费金额": np.round(total_spend, 2),
        "最近消费金额": np.round(recent_spend, 2),
        "消费频率": np.round(purchase_frequency, 2),
        "客单价": np.round(avg_order_value, 2),
        "评论数": comments,
        "分享数": shares,
        "点赞数": likes,
        "反馈次数": feedbacks,
        "客服咨询次数": service_contacts,
        "投诉次数": complaints,
        "评分均值": np.round(rating, 2),
        "是否流失": churn,
    })

    # 少量缺失值，用于测试预处理流程。
    for col in ["累计消费金额", "最近消费金额", "活跃频率", "评分均值", "客服咨询次数"]:
        mask = rng.random(n_samples) < 0.018
        df.loc[mask, col] = np.nan

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"数据集已生成: {output_path}")
    print(f"样本数: {len(df)}")
    print(f"流失比例: {df['是否流失'].mean():.2%}")
    return df


if __name__ == "__main__":
    generate_dataset()
