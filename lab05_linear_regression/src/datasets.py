import numpy as np


def generate_data(m=100, noise_std=2, seed=42):
    """
    生成模拟数据
    y = 10 + 5x + noise
    """
    np.random.seed(seed)
    x = np.linspace(1, 10, m).reshape(-1, 1)
    noise = np.random.normal(0, noise_std, size=(m, 1))
    y = 10 + 5 * x + noise
    return x, y


def train_test_split_simple(X, Y, test_size=0.2, seed=42):
    """
    手动划分训练集和测试集
    """
    np.random.seed(seed)
    m = X.shape[0]
    indices = np.arange(m)
    np.random.shuffle(indices)

    test_count = int(m * test_size)
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]

    X_train = X[train_idx]
    X_test = X[test_idx]
    Y_train = Y[train_idx]
    Y_test = Y[test_idx]

    return X_train, X_test, Y_train, Y_test


def standardize_train_test(X_train, X_test):
    """
    用训练集的均值和标准差做标准化，避免数据泄露
    """
    mu = X_train.mean()
    sigma = X_train.std()

    X_train_scaled = (X_train - mu) / sigma
    X_test_scaled = (X_test - mu) / sigma

    return X_train_scaled, X_test_scaled, mu, sigma
