import os
import numpy as np
import pandas as pd


def generate_gym_user_data(n_high=100, n_low=100, n_churn=100, random_state=42):
    """
    生成健身房用户行为模拟数据集。

    标签含义：
    0：高复购用户
    1：低活跃用户
    2：流失风险用户
    """
    np.random.seed(random_state)

    rows = []

    # 0 高复购用户：最近消费天数少、月消费频率高、金额较高、到店时长较长、常参加团课
    for _ in range(n_high):
        rows.append({
            "最近一次消费天数": np.random.randint(1, 16),
            "月消费频率": np.random.randint(5, 13),
            "平均单次消费金额": round(np.random.normal(180, 35), 2),
            "单次平均到店时长": round(np.random.normal(70, 12), 2),
            "是否参与团课": np.random.choice(["是", "否"], p=[0.8, 0.2]),
            "会员等级": np.random.choice(["黄金", "铂金", "白银"], p=[0.5, 0.3, 0.2]),
            "标签": 0
        })

    # 1 低活跃用户：最近消费天数中等、月消费频率较低、金额中等、到店时长中短
    for _ in range(n_low):
        rows.append({
            "最近一次消费天数": np.random.randint(15, 36),
            "月消费频率": np.random.randint(2, 6),
            "平均单次消费金额": round(np.random.normal(120, 25), 2),
            "单次平均到店时长": round(np.random.normal(45, 10), 2),
            "是否参与团课": np.random.choice(["是", "否"], p=[0.35, 0.65]),
            "会员等级": np.random.choice(["普通", "白银", "黄金"], p=[0.45, 0.4, 0.15]),
            "标签": 1
        })

    # 2 流失风险用户：最近消费天数长、月消费频率低、金额偏低、到店时长短、很少参加团课
    for _ in range(n_churn):
        rows.append({
            "最近一次消费天数": np.random.randint(35, 91),
            "月消费频率": np.random.randint(0, 3),
            "平均单次消费金额": round(np.random.normal(80, 20), 2),
            "单次平均到店时长": round(np.random.normal(28, 8), 2),
            "是否参与团课": np.random.choice(["是", "否"], p=[0.15, 0.85]),
            "会员等级": np.random.choice(["普通", "白银"], p=[0.75, 0.25]),
            "标签": 2
        })

    data = pd.DataFrame(rows)

    # 限制数值范围，避免出现负数或不合理值
    data["平均单次消费金额"] = data["平均单次消费金额"].clip(lower=30, upper=300)
    data["单次平均到店时长"] = data["单次平均到店时长"].clip(lower=10, upper=120)

    # 随机打乱数据
    data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # 人为加入少量缺失值，用于体现 C4.5 对缺失值的处理能力
    missing_cols = ["平均单次消费金额", "单次平均到店时长", "是否参与团课", "会员等级"]
    for col in missing_cols:
        missing_index = data.sample(frac=0.05, random_state=random_state + len(col)).index
        data.loc[missing_index, col] = np.nan

    return data


def main():
    os.makedirs("data", exist_ok=True)

    data = generate_gym_user_data(
        n_high=100,
        n_low=100,
        n_churn=100,
        random_state=42
    )

    output_path = "data/gym_user_data.csv"
    data.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("模拟数据集已生成：", output_path)
    print("数据集形状：", data.shape)
    print()
    print("各类别数量：")
    print(data["标签"].value_counts().sort_index())
    print()
    print("前 5 行数据：")
    print(data.head())


if __name__ == "__main__":
    main()
