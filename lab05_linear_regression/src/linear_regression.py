import numpy as np


def add_bias(X):
    """
    给特征添加偏置项，对应 theta0
    """
    return np.c_[np.ones((X.shape[0], 1)), X]


def gradient_descent_visual(X_train, Y_train, alpha=0.1, max_iter=10000, tol=1e-6, record_interval=50):
    """
    批量梯度下降（带参数历史记录）
    """
    m = X_train.shape[0]
    X_b = add_bias(X_train)

    theta = np.zeros((2, 1))
    loss_history = []
    theta_history = []

    for i in range(max_iter):
        # 预测
        Y_pred = X_b @ theta

        # 损失函数
        loss = (1 / (2 * m)) * np.sum((Y_pred - Y_train) ** 2)
        loss_history.append(loss)

        # 记录参数
        if i % record_interval == 0:
            theta_history.append(theta.copy())

        # 梯度
        gradient = (1 / m) * X_b.T @ (Y_pred - Y_train)

        # 更新参数
        theta = theta - alpha * gradient

        # 收敛判断
        if i > 0 and abs(loss_history[i] - loss_history[i - 1]) < tol:
            print(f"模型在第 {i} 次迭代收敛")
            theta_history.append(theta.copy())
            break

    return theta, loss_history, theta_history


def normal_equation(X_train, Y_train):
    """
    正规方程法
    """
    X_b = add_bias(X_train)
    theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ Y_train
    return theta


def predict(X, theta):
    """
    预测函数
    """
    X_b = add_bias(X)
    return X_b @ theta


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def r2_score_manual(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot