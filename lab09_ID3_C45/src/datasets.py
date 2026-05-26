import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(path):
    """
    读取健身房用户行为数据集。
    """
    data = pd.read_csv(path)
    return data


def fill_missing_for_id3(data):
    """
    ID3 不能直接处理缺失值，因此这里先进行基础填充。
    连续特征使用均值填充，离散特征使用众数填充。
    """
    data = data.copy()

    for col in data.columns:
        if col == "标签":
            continue

        if data[col].dtype in ["int64", "float64"]:
            data[col] = data[col].fillna(data[col].mean())
        else:
            data[col] = data[col].fillna(data[col].mode()[0])

    return data


def discretize_for_id3(data):
    """
    ID3 不能直接处理连续特征，所以需要按业务逻辑进行分桶。
    """
    data = data.copy()

    data["最近一次消费天数"] = pd.cut(
        data["最近一次消费天数"],
        bins=[-1, 14, 30, float("inf")],
        labels=["近期消费", "中期未消费", "长期未消费"]
    )

    data["月消费频率"] = pd.cut(
        data["月消费频率"],
        bins=[-1, 2, 5, float("inf")],
        labels=["低频", "中频", "高频"]
    )

    data["平均单次消费金额"] = pd.cut(
        data["平均单次消费金额"],
        bins=[-1, 100, 200, float("inf")],
        labels=["低消费", "中消费", "高消费"]
    )

    data["单次平均到店时长"] = pd.cut(
        data["单次平均到店时长"],
        bins=[-1, 30, 60, float("inf")],
        labels=["短时长", "中等时长", "长时长"]
    )

    return data


def split_data(data, label_col="标签", test_size=0.3, random_state=42):
    """
    使用分层抽样划分训练集和测试集，保证三类用户比例一致。
    """
    X = data.drop(columns=[label_col])
    y = data[label_col].astype(int)

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
