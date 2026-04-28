import os
import numpy as np
import matplotlib.pyplot as plt
import chineseize_matplotlib

from datasets import generate_data, train_test_split_simple, standardize_train_test
from linear_regression import (
    gradient_descent_visual,
    normal_equation,
    predict,
    mse,
    rmse,
    mae,
    r2_score_manual,
)

# 图片输出目录（相对路径）
FIG_DIR = "../figures/figures"


def ensure_figure_dir():
    os.makedirs(FIG_DIR, exist_ok=True)


def plot_scatter(X, Y):
    plt.figure(figsize=(10, 6))
    plt.scatter(X, Y, edgecolor="k", alpha=0.7, s=60, label="学生样本")
    plt.xlabel("每周学习时长（小时）")
    plt.ylabel("考试成绩（分）")
    plt.title("学习时长与考试成绩的散点图")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(FIG_DIR, "scatter.png"), dpi=300, bbox_inches="tight")
    plt.show()


def plot_fit_process(X_train, Y_train, theta_history, mu, sigma, alpha):
    x_line = np.linspace(0, 12, 100).reshape(-1, 1)
    x_line_scaled = (x_line - mu) / sigma
    x_line_b = np.c_[np.ones((100, 1)), x_line_scaled]

    plt.figure(figsize=(12, 7))
    plt.scatter(X_train, Y_train, edgecolor="k", alpha=0.6, s=50, label="训练集样本")

    colors = plt.cm.viridis(np.linspace(0, 1, len(theta_history)))
    for i, (theta, color) in enumerate(zip(theta_history, colors)):
        y_line = x_line_b @ theta
        if i == 0:
            plt.plot(
                x_line,
                y_line,
                color=color,
                linestyle="--",
                linewidth=1.5,
                label="初始状态",
            )
        elif i == len(theta_history) - 1:
            plt.plot(
                x_line, y_line, color="darkorange", linewidth=3, label="最终拟合直线"
            )
        else:
            plt.plot(x_line, y_line, color=color, linestyle=":", linewidth=1.2)

    y_true_line = 10 + 5 * x_line
    plt.plot(
        x_line, y_true_line, color="red", linestyle="--", linewidth=2, label="真实直线"
    )

    plt.xlabel("每周学习时长（小时）")
    plt.ylabel("考试成绩（分）")
    plt.title(f"梯度下降法拟合直线的迭代过程（alpha={alpha}）")
    plt.xlim(0, 11)
    plt.ylim(5, 65)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(FIG_DIR, "fit_process.png"), dpi=300, bbox_inches="tight")
    plt.show()


def plot_loss_curve(loss_history, alpha):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, linewidth=2)
    plt.xlabel("迭代次数")
    plt.ylabel("损失函数值（Loss）")
    plt.title(f"梯度下降法损失函数收敛曲线（alpha={alpha}）")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(FIG_DIR, "loss_curve.png"), dpi=300, bbox_inches="tight")
    plt.show()


def plot_learning_rate_compare(X_train_scaled, Y_train):
    alphas = [0.001, 0.01, 0.1, 0.5]
    loss_histories = []

    for alpha in alphas:
        print(f"\n正在训练 alpha={alpha} 的模型...")
        _, loss_hist, _ = gradient_descent_visual(
            X_train_scaled, Y_train, alpha=alpha, record_interval=100
        )
        loss_histories.append((alpha, loss_hist))

    plt.figure(figsize=(12, 7))
    for alpha, loss_hist in loss_histories:
        plt.plot(loss_hist, linewidth=2, label=f"alpha={alpha}", alpha=0.8)

    plt.xlabel("迭代次数")
    plt.ylabel("损失函数值（Loss）")
    plt.title("不同学习率下损失函数收敛曲线对比")
    plt.xlim(0, 500)
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(
        os.path.join(FIG_DIR, "learning_rate_compare.png"), dpi=300, bbox_inches="tight"
    )
    plt.show()


def plot_final_fit(X_train, Y_train, X_test, Y_test, theta_bgd, mu, sigma):
    x_line = np.linspace(0, 12, 100).reshape(-1, 1)
    x_line_scaled = (x_line - mu) / sigma
    y_line = predict(x_line_scaled, theta_bgd)

    plt.figure(figsize=(12, 7))
    plt.scatter(X_train, Y_train, edgecolor="k", alpha=0.6, s=60, label="训练集样本")
    plt.scatter(
        X_test, Y_test, edgecolor="k", alpha=0.9, s=70, marker="s", label="测试集样本"
    )
    plt.plot(x_line, y_line, color="darkorange", linewidth=3, label="最终拟合直线")
    plt.plot(
        x_line,
        10 + 5 * x_line,
        color="red",
        linestyle="--",
        linewidth=2,
        label="真实直线",
    )

    plt.xlabel("每周学习时长（小时）")
    plt.ylabel("考试成绩（分）")
    plt.title("一元线性回归最终拟合效果图")
    plt.xlim(0, 11)
    plt.ylim(5, 65)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(FIG_DIR, "final_fit.png"), dpi=300, bbox_inches="tight")
    plt.show()


def main():
    ensure_figure_dir()

    # 1. 生成数据
    X, Y = generate_data(m=100, noise_std=2, seed=42)

    print("=" * 50)
    print("数据集探索性分析")
    print("=" * 50)
    print(f"样本总数：{X.shape[0]}")
    print(f"学习时长范围：{X.min():.1f}小时 ~ {X.max():.1f}小时")
    print(f"考试成绩范围：{Y.min():.1f}分 ~ {Y.max():.1f}分")
    print("真实参数：theta0=10, theta1=5")

    plot_scatter(X, Y)

    # 2. 划分数据集
    X_train, X_test, Y_train, Y_test = train_test_split_simple(
        X, Y, test_size=0.2, seed=42
    )

    print("=" * 50)
    print("数据集划分结果")
    print("=" * 50)
    print(f"训练集样本数：{X_train.shape[0]}")
    print(f"测试集样本数：{X_test.shape[0]}")

    # 3. 标准化
    X_train_scaled, X_test_scaled, mu, sigma = standardize_train_test(X_train, X_test)

    print("=" * 50)
    print("特征标准化信息")
    print("=" * 50)
    print(f"训练集均值 mu：{mu:.4f}")
    print(f"训练集标准差 sigma：{sigma:.4f}")

    # 4. 梯度下降训练
    alpha = 0.1
    theta_bgd, loss_history, theta_history = gradient_descent_visual(
        X_train_scaled, Y_train, alpha=alpha, record_interval=50
    )

    print("=" * 50)
    print("梯度下降法训练结果")
    print("=" * 50)
    print(f"theta0（截距）：{theta_bgd[0, 0]:.4f}")
    print(f"theta1（斜率）：{theta_bgd[1, 0]:.4f}")

    # 5. 正规方程
    theta_normal = normal_equation(X_train_scaled, Y_train)

    print("=" * 50)
    print("正规方程法 vs 梯度下降法")
    print("=" * 50)
    print(f"{'方法':<15} | {'theta0':<12} | {'theta1':<12}")
    print("-" * 45)
    print(
        f"{'正规方程法':<15} | {theta_normal[0,0]:<12.4f} | {theta_normal[1,0]:<12.4f}"
    )
    print(f"{'梯度下降法':<15} | {theta_bgd[0,0]:<12.4f} | {theta_bgd[1,0]:<12.4f}")

    # 6. 测试集预测与评估
    Y_pred = predict(X_test_scaled, theta_bgd)

    mse_value = mse(Y_test, Y_pred)
    rmse_value = rmse(Y_test, Y_pred)
    mae_value = mae(Y_test, Y_pred)
    r2_value = r2_score_manual(Y_test, Y_pred)

    print("=" * 50)
    print("模型在测试集上的评估结果")
    print("=" * 50)
    print(f"MSE ：{mse_value:.4f}")
    print(f"RMSE：{rmse_value:.4f}")
    print(f"MAE ：{mae_value:.4f}")
    print(f"R²  ：{r2_value:.4f}")

    # 7. 画图
    plot_fit_process(X_train, Y_train, theta_history, mu, sigma, alpha)
    plot_loss_curve(loss_history, alpha)
    plot_learning_rate_compare(X_train_scaled, Y_train)
    plot_final_fit(X_train, Y_train, X_test, Y_test, theta_bgd, mu, sigma)


if __name__ == "__main__":
    main()
