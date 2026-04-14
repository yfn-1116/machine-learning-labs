from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

from datasets import generate_abtest_data, standardize_days
from lwlr import ols_fit_predict, lwlr_predict_all, lowess


def ensure_figure_dir() -> Path:
    """
    确保 figures 文件夹存在
    """
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    figure_dir = project_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir


def plot_raw_and_smooth(
    days: np.ndarray,
    raw_control: np.ndarray,
    raw_exp: np.ndarray,
    smooth_control: np.ndarray,
    smooth_exp: np.ndarray,
    save_path: Path,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(days, raw_control, label="Control Raw", alpha=0.8)
    plt.scatter(days, raw_exp, label="Experiment Raw", alpha=0.8)
    plt.plot(days, smooth_control, linewidth=2.2, label="Control LOWESS")
    plt.plot(days, smooth_exp, linewidth=2.2, label="Experiment LOWESS")
    plt.xlabel("Day")
    plt.ylabel("Payment Rate")
    plt.title("Raw Data vs LOWESS Smoothed Curves")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_ols_vs_lwlr(
    days: np.ndarray,
    raw_y: np.ndarray,
    ols_y: np.ndarray,
    lwlr_y: np.ndarray,
    save_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(days, raw_y, label="Raw Data", alpha=0.85)
    plt.plot(days, ols_y, linewidth=2.0, label="OLS")
    plt.plot(days, lwlr_y, linewidth=2.2, label="LWLR")
    plt.xlabel("Day")
    plt.ylabel("Payment Rate")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_bandwidth_compare(
    days: np.ndarray,
    raw_y: np.ndarray,
    smooth_01: np.ndarray,
    smooth_03: np.ndarray,
    smooth_05: np.ndarray,
    save_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(days, raw_y, label="Raw Data", alpha=0.75)
    plt.plot(days, smooth_01, linewidth=2.0, label="LOWESS frac=0.1")
    plt.plot(days, smooth_03, linewidth=2.0, label="LOWESS frac=0.3")
    plt.plot(days, smooth_05, linewidth=2.0, label="LOWESS frac=0.5")
    plt.xlabel("Day")
    plt.ylabel("Payment Rate")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main() -> None:
    figure_dir = ensure_figure_dir()

    # 1. 数据准备
    df = generate_abtest_data(random_seed=42)

    days = df["day"].to_numpy()
    x_std = standardize_days(days).reshape(-1, 1)

    y_control = df["control"].to_numpy()
    y_exp = df["experiment"].to_numpy()

    # 2. OLS
    ols_control = ols_fit_predict(x_std, y_control, x_std)
    ols_exp = ols_fit_predict(x_std, y_exp, x_std)

    # 3. LWLR
    lwlr_control = lwlr_predict_all(x_std, y_control, tau=0.15)
    lwlr_exp = lwlr_predict_all(x_std, y_exp, tau=0.15)

    # 4. LOWESS 原生实现
    lowess_control_01 = lowess(days, y_control, frac=0.1, it=3)
    lowess_control_03 = lowess(days, y_control, frac=0.3, it=3)
    lowess_control_05 = lowess(days, y_control, frac=0.5, it=3)

    lowess_exp_01 = lowess(days, y_exp, frac=0.1, it=3)
    lowess_exp_03 = lowess(days, y_exp, frac=0.3, it=3)
    lowess_exp_05 = lowess(days, y_exp, frac=0.5, it=3)

    # 5. statsmodels 验证
    sm_control = sm_lowess(y_control, days, frac=0.3, return_sorted=False)
    sm_exp = sm_lowess(y_exp, days, frac=0.3, return_sorted=False)

    # 6. 保存图片到 figures/
    plot_raw_and_smooth(
        days,
        y_control,
        y_exp,
        lowess_control_03,
        lowess_exp_03,
        figure_dir / "raw_vs_lowess.png",
    )

    plot_ols_vs_lwlr(
        days,
        y_control,
        ols_control,
        lwlr_control,
        figure_dir / "control_ols_vs_lwlr.png",
        "Control Group: OLS vs LWLR",
    )

    plot_ols_vs_lwlr(
        days,
        y_exp,
        ols_exp,
        lwlr_exp,
        figure_dir / "experiment_ols_vs_lwlr.png",
        "Experiment Group: OLS vs LWLR",
    )

    plot_bandwidth_compare(
        days,
        y_control,
        lowess_control_01,
        lowess_control_03,
        lowess_control_05,
        figure_dir / "control_bandwidth_compare.png",
        "Control Group: LOWESS Bandwidth Comparison",
    )

    plot_bandwidth_compare(
        days,
        y_exp,
        lowess_exp_01,
        lowess_exp_03,
        lowess_exp_05,
        figure_dir / "experiment_bandwidth_compare.png",
        "Experiment Group: LOWESS Bandwidth Comparison",
    )

    # 7. 控制台输出关键结果
    mae_control = np.mean(np.abs(lowess_control_03 - sm_control))
    mae_exp = np.mean(np.abs(lowess_exp_03 - sm_exp))

    print("=" * 60)
    print("实验七：LWLR + LOWESS 原生实现")
    print("=" * 60)
    print(f"原生 LOWESS 与 statsmodels 的控制组平均绝对误差: {mae_control:.8f}")
    print(f"原生 LOWESS 与 statsmodels 的实验组平均绝对误差: {mae_exp:.8f}")
    print()
    print("图片已保存到 figures 文件夹：")
    print(f"- {figure_dir / 'raw_vs_lowess.png'}")
    print(f"- {figure_dir / 'control_ols_vs_lwlr.png'}")
    print(f"- {figure_dir / 'experiment_ols_vs_lwlr.png'}")
    print(f"- {figure_dir / 'control_bandwidth_compare.png'}")
    print(f"- {figure_dir / 'experiment_bandwidth_compare.png'}")


if __name__ == "__main__":
    main()