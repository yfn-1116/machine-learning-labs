# -*- coding: utf-8 -*-
"""
experiment.py
完整岭回归实验流程：
- 生成/读取数据
- 缺失值处理
- IQR异常值裁剪
- 划分训练集测试集
- 标准化
- 多重共线性分析
- VIF
- OLS训练与评估
- Ridge训练与评估
- 5折交叉验证选择最优lambda
- 输出性能表、系数表、特征重要性表
- 生成并保存所有图表到 figures/
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import chineseize_matplotlib  # type: ignore  # noqa: F401

    _CHINESEIZE_AVAILABLE = True
except Exception:
    _CHINESEIZE_AVAILABLE = False

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from datasets import generate_sales_dataset, load_dataset, TARGET_NAME
from ridge import (
    ols_fit,
    ridge_fit,
    predict,
    regression_metrics,
    ridge_trace,
)


warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True, precision=4)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 200)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "figures"
DATA_PATH = FIG_DIR / "新品销量预测数据集.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.3


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_matplotlib_for_chinese() -> None:
    font_candidates = [
        "SimHei",
        "Microsoft YaHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
        "PingFang SC",
        "Heiti SC",
        "STHeiti",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["font.sans-serif"] = font_candidates
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["figure.dpi"] = 120
    matplotlib.rcParams["savefig.dpi"] = 150

    if _CHINESEIZE_AVAILABLE:
        print("[INFO] 已启用 chineseize_matplotlib（若环境支持）。")
    else:
        print("[INFO] 未安装 chineseize_matplotlib，已启用中文字体兜底方案。")


def save_current_figure(save_path: Path) -> None:
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    plt.tight_layout()
    plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.10)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.2)
    print(f"[OK] 已保存: {save_path}")


def maybe_generate_or_load_data() -> Tuple[pd.DataFrame, np.ndarray]:
    ensure_dir(FIG_DIR)
    if DATA_PATH.exists():
        print(f"[INFO] 检测到已有数据集，直接读取: {DATA_PATH}")
        df = load_dataset(DATA_PATH)
        _, true_beta = generate_sales_dataset(
            n_samples=50,
            random_state=RANDOM_STATE,
            save_csv=False,
            add_missing_ratio=0.0,
            add_outlier_ratio=0.0,
        )
        return df, true_beta

    print("[INFO] 未发现现成数据集，开始生成模拟数据集...")
    df, true_beta = generate_sales_dataset(
        n_samples=50,
        random_state=RANDOM_STATE,
        save_csv=True,
        csv_path=DATA_PATH,
        add_missing_ratio=0.03,
        add_outlier_ratio=0.02,
    )
    return df, true_beta


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[STEP] 缺失值处理")
    missing_count = df.isnull().sum()
    print("缺失值统计：")
    print(missing_count)
    df_filled = df.copy()
    for col in df_filled.columns:
        if df_filled[col].isnull().any():
            median_val = df_filled[col].median()
            df_filled[col] = df_filled[col].fillna(median_val)
            print(f"[INFO] 已用中位数填充缺失值: {col} -> {median_val:.4f}")
    return df_filled


def clip_outliers_iqr(
    df: pd.DataFrame, exclude_cols: List[str] | None = None
) -> pd.DataFrame:
    print("\n[STEP] IQR 异常值裁剪")
    exclude_cols = exclude_cols or []
    df_clipped = df.copy()

    for col in df_clipped.columns:
        if col in exclude_cols:
            continue
        q1 = df_clipped[col].quantile(0.25)
        q3 = df_clipped[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        before_outliers = ((df_clipped[col] < lower) | (df_clipped[col] > upper)).sum()
        df_clipped[col] = df_clipped[col].clip(lower=lower, upper=upper)
        if before_outliers > 0:
            print(f"[INFO] {col} 已裁剪异常值数量: {int(before_outliers)}")
    return df_clipped


def split_and_scale(df: pd.DataFrame):
    print("\n[STEP] 划分训练集与测试集，并进行标准化")

    X = df.drop(columns=[TARGET_NAME])
    y = df[TARGET_NAME].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    std_check = X_train.std(numeric_only=True)
    const_cols = std_check[std_check == 0].index.tolist()
    if const_cols:
        raise ValueError(f"训练集中存在常数特征，无法标准化: {const_cols}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled_df = pd.DataFrame(
        X_train_scaled, columns=X.columns, index=X_train.index
    )
    X_test_scaled_df = pd.DataFrame(
        X_test_scaled, columns=X.columns, index=X_test.index
    )

    print("[INFO] 训练集形状:", X_train.shape)
    print("[INFO] 测试集形状:", X_test.shape)
    print(
        "[INFO] 训练集标准化后均值（前5个）:",
        np.round(X_train_scaled_df.mean().values[:5], 6),
    )
    print(
        "[INFO] 训练集标准化后标准差（前5个）:",
        np.round(X_train_scaled_df.std(ddof=0).values[:5], 6),
    )

    return (
        X,
        y,
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled_df,
        X_test_scaled_df,
        scaler,
    )


def compute_vif(X_scaled_df: pd.DataFrame) -> pd.DataFrame:
    print("\n[STEP] 多重共线性分析：VIF")
    vif_df = pd.DataFrame(
        {
            "特征名称": X_scaled_df.columns,
            "VIF值": [
                variance_inflation_factor(X_scaled_df.values, i)
                for i in range(X_scaled_df.shape[1])
            ],
        }
    )
    vif_df = vif_df.sort_values("VIF值", ascending=False).reset_index(drop=True)
    print(vif_df)
    return vif_df


def plot_corr_heatmap(X_train_df: pd.DataFrame) -> None:
    print("\n[STEP] 绘制特征相关性热力图")
    corr_matrix = X_train_df.corr()

    plt.figure(figsize=(14, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=False,
        cbar=True,
        annot_kws={"size": 8},
    )
    plt.title("特征相关性热力图", fontsize=16, pad=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    save_current_figure(FIG_DIR / "特征相关性热力图.png")
    plt.close()


def run_ols(X_train_scaled_df, X_test_scaled_df, y_train, y_test):
    print("\n[STEP] OLS 训练与评估")
    ols_result = {
        "success": False,
        "coef": None,
        "intercept": 0.0,
        "train_metrics": None,
        "test_metrics": None,
        "y_test_pred": None,
        "error": None,
    }

    try:
        coef_, intercept_ = ols_fit(
            X_train_scaled_df.values, y_train.values, add_intercept=True
        )
        y_train_pred = predict(X_train_scaled_df.values, coef_, intercept_)
        y_test_pred = predict(X_test_scaled_df.values, coef_, intercept_)

        train_metrics = regression_metrics(y_train.values, y_train_pred)
        test_metrics = regression_metrics(y_test.values, y_test_pred)

        ols_result.update(
            {
                "success": True,
                "coef": coef_,
                "intercept": intercept_,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "y_test_pred": y_test_pred,
            }
        )

        print("[OK] OLS 训练成功")
        print("[INFO] OLS 训练集指标:", train_metrics)
        print("[INFO] OLS 测试集指标:", test_metrics)

    except Exception as exc:
        ols_result["error"] = str(exc)
        print(f"[WARN] OLS 训练失败，但程序会继续执行 Ridge 部分: {exc}")

    return ols_result


def cross_validate_ridge(
    X_train_scaled_df, y_train, lambdas: np.ndarray, n_splits: int = 5
):
    print("\n[STEP] 5 折交叉验证选择最优 lambda")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    cv_records = []

    X_values = X_train_scaled_df.values
    y_values = y_train.values

    for lam in lambdas:
        fold_mse = []
        for train_idx, val_idx in kf.split(X_values):
            X_tr, X_val = X_values[train_idx], X_values[val_idx]
            y_tr, y_val = y_values[train_idx], y_values[val_idx]

            coef_, intercept_ = ridge_fit(X_tr, y_tr, lam=lam, add_intercept=True)
            y_val_pred = predict(X_val, coef_, intercept_)
            mse_val = np.mean((y_val - y_val_pred) ** 2)
            fold_mse.append(mse_val)

        cv_records.append(
            {
                "lambda": lam,
                "cv_mse_mean": float(np.mean(fold_mse)),
                "cv_mse_std": float(np.std(fold_mse)),
            }
        )

    cv_df = pd.DataFrame(cv_records)
    best_row = cv_df.loc[cv_df["cv_mse_mean"].idxmin()]
    best_lambda = float(best_row["lambda"])

    print(f"[OK] 交叉验证最优 lambda = {best_lambda:.6f}")
    print("[INFO] 最低平均 CV-MSE =", round(float(best_row["cv_mse_mean"]), 4))

    return best_lambda, cv_df


def plot_ridge_trace(
    X_train_scaled_df, y_train, feature_names: List[str], lambdas_trace: np.ndarray
):
    print("\n[STEP] 绘制岭迹图")
    coef_path = ridge_trace(
        X_train_scaled_df.values, y_train.values, lambdas_trace, add_intercept=True
    )

    plt.figure(figsize=(16, 10))
    for i, name in enumerate(feature_names):
        plt.plot(lambdas_trace, coef_path[:, i], linewidth=1.5, label=name)

    plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("正则化参数 λ", fontsize=13)
    plt.ylabel("回归系数", fontsize=13)
    plt.title("岭迹图", fontsize=16, pad=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=True)

    save_current_figure(FIG_DIR / "岭迹图.png")
    plt.close()

    coef_df = pd.DataFrame(coef_path, columns=feature_names)
    coef_df.insert(0, "lambda", lambdas_trace)
    coef_df.to_csv(FIG_DIR / "岭迹图系数轨迹.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] 已保存: {FIG_DIR / '岭迹图系数轨迹.csv'}")


def plot_cv_curve(cv_df: pd.DataFrame, best_lambda: float):
    print("\n[STEP] 绘制交叉验证 MSE 曲线")
    plt.figure(figsize=(10, 6))
    plt.plot(cv_df["lambda"], cv_df["cv_mse_mean"], linewidth=2)
    plt.scatter(
        [best_lambda],
        [cv_df.loc[cv_df["lambda"] == best_lambda, "cv_mse_mean"].values[0]],
        s=80,
        label=f"最优 λ = {best_lambda:.4f}",
    )
    plt.xscale("log")
    plt.xlabel("正则化参数 λ（对数刻度）", fontsize=13)
    plt.ylabel("5 折交叉验证平均 MSE", fontsize=13)
    plt.title("交叉验证 MSE 曲线", fontsize=16, pad=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_current_figure(FIG_DIR / "交叉验证MSE曲线.png")
    plt.close()


def run_ridge(X_train_scaled_df, X_test_scaled_df, y_train, y_test, best_lambda: float):
    print("\n[STEP] Ridge 训练与评估")
    coef_, intercept_ = ridge_fit(
        X_train_scaled_df.values, y_train.values, lam=best_lambda, add_intercept=True
    )
    y_train_pred = predict(X_train_scaled_df.values, coef_, intercept_)
    y_test_pred = predict(X_test_scaled_df.values, coef_, intercept_)

    train_metrics = regression_metrics(y_train.values, y_train_pred)
    test_metrics = regression_metrics(y_test.values, y_test_pred)

    print("[OK] Ridge 训练成功")
    print("[INFO] Ridge 训练集指标:", train_metrics)
    print("[INFO] Ridge 测试集指标:", test_metrics)

    return {
        "coef": coef_,
        "intercept": intercept_,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "y_test_pred": y_test_pred,
    }


def plot_true_vs_pred(y_test, y_pred, model_name: str, save_name: str):
    print(f"\n[STEP] 绘制测试集真实值 vs 预测值散点图（{model_name}）")
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.75)
    min_val = min(np.min(y_test), np.min(y_pred))
    max_val = max(np.max(y_test), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=1.5)
    plt.xlabel("测试集真实值", fontsize=13)
    plt.ylabel("测试集预测值", fontsize=13)
    plt.title(f"测试集真实值 vs 预测值散点图（{model_name}）", fontsize=15, pad=12)
    plt.grid(True, alpha=0.3)
    save_current_figure(FIG_DIR / save_name)
    plt.close()


def plot_residual_distribution(y_test, y_pred, model_name: str, save_name: str):
    print(f"\n[STEP] 绘制测试集残差分布图（{model_name}）")
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, bins=12, kde=True)
    plt.axvline(0, linestyle="--", linewidth=1.5)
    plt.xlabel("残差", fontsize=13)
    plt.ylabel("频数", fontsize=13)
    plt.title(f"测试集残差分布图（{model_name}）", fontsize=15, pad=12)
    plt.grid(True, alpha=0.25)
    save_current_figure(FIG_DIR / save_name)
    plt.close()


def plot_coef_bar(
    feature_names: List[str], coefs: np.ndarray, title: str, save_name: str
):
    coef_df = pd.DataFrame(
        {
            "特征名称": feature_names,
            "回归系数": coefs,
        }
    ).sort_values("回归系数", ascending=False)

    plt.figure(figsize=(12, 7))
    sns.barplot(data=coef_df, x="回归系数", y="特征名称", orient="h")
    plt.xlabel("回归系数", fontsize=13)
    plt.ylabel("特征名称", fontsize=13)
    plt.title(title, fontsize=15, pad=12)
    plt.grid(True, axis="x", alpha=0.3)
    save_current_figure(FIG_DIR / save_name)
    plt.close()


def save_tables(
    feature_names: List[str],
    true_beta: np.ndarray,
    vif_df: pd.DataFrame,
    ols_result: Dict,
    ridge_result: Dict,
    best_lambda: float,
) -> None:
    print("\n[STEP] 输出性能对比表、系数对比表、特征重要性表")

    rows = []

    if ols_result["success"]:
        rows.append(
            {
                "模型": "OLS",
                "训练集MSE": ols_result["train_metrics"]["MSE"],
                "训练集MAE": ols_result["train_metrics"]["MAE"],
                "训练集R2": ols_result["train_metrics"]["R2"],
                "测试集MSE": ols_result["test_metrics"]["MSE"],
                "测试集MAE": ols_result["test_metrics"]["MAE"],
                "测试集R2": ols_result["test_metrics"]["R2"],
            }
        )
    else:
        rows.append(
            {
                "模型": "OLS",
                "训练集MSE": np.nan,
                "训练集MAE": np.nan,
                "训练集R2": np.nan,
                "测试集MSE": np.nan,
                "测试集MAE": np.nan,
                "测试集R2": np.nan,
            }
        )

    rows.append(
        {
            "模型": f"Ridge(lambda={best_lambda:.6f})",
            "训练集MSE": ridge_result["train_metrics"]["MSE"],
            "训练集MAE": ridge_result["train_metrics"]["MAE"],
            "训练集R2": ridge_result["train_metrics"]["R2"],
            "测试集MSE": ridge_result["test_metrics"]["MSE"],
            "测试集MAE": ridge_result["test_metrics"]["MAE"],
            "测试集R2": ridge_result["test_metrics"]["R2"],
        }
    )

    performance_df = pd.DataFrame(rows)
    print("\n性能对比表：")
    print(performance_df.round(4))
    performance_df.to_csv(FIG_DIR / "性能对比表.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] 已保存: {FIG_DIR / '性能对比表.csv'}")

    coef_compare_df = pd.DataFrame(
        {
            "特征名称": feature_names,
            "真实系数": true_beta,
            "Ridge系数": ridge_result["coef"],
            "Ridge系数绝对误差": np.abs(ridge_result["coef"] - true_beta),
        }
    )

    if ols_result["success"]:
        coef_compare_df["OLS系数"] = ols_result["coef"]
        coef_compare_df["OLS系数绝对误差"] = np.abs(ols_result["coef"] - true_beta)
    else:
        coef_compare_df["OLS系数"] = np.nan
        coef_compare_df["OLS系数绝对误差"] = np.nan

    coef_compare_df = coef_compare_df[
        [
            "特征名称",
            "真实系数",
            "OLS系数",
            "Ridge系数",
            "OLS系数绝对误差",
            "Ridge系数绝对误差",
        ]
    ]
    print("\n系数对比表：")
    print(coef_compare_df.round(4))
    coef_compare_df.to_csv(
        FIG_DIR / "系数对比表.csv", index=False, encoding="utf-8-sig"
    )
    print(f"[OK] 已保存: {FIG_DIR / '系数对比表.csv'}")

    importance_df = pd.DataFrame(
        {
            "特征名称": feature_names,
            "Ridge系数": ridge_result["coef"],
            "重要性(|Ridge系数|)": np.abs(ridge_result["coef"]),
            "VIF值": vif_df.set_index("特征名称").loc[feature_names, "VIF值"].values,
        }
    ).sort_values("重要性(|Ridge系数|)", ascending=False)

    print("\n特征重要性表：")
    print(importance_df.round(4))
    importance_df.to_csv(
        FIG_DIR / "特征重要性表.csv", index=False, encoding="utf-8-sig"
    )
    print(f"[OK] 已保存: {FIG_DIR / '特征重要性表.csv'}")

    vif_df.to_csv(FIG_DIR / "VIF表.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] 已保存: {FIG_DIR / 'VIF表.csv'}")


def main():
    print("=" * 80)
    print("岭回归实验：新品销量预测（原生实现）")
    print("=" * 80)

    ensure_dir(FIG_DIR)
    configure_matplotlib_for_chinese()

    df_raw, true_beta = maybe_generate_or_load_data()
    print("\n[INFO] 原始数据预览：")
    print(df_raw.head())

    df_filled = fill_missing_values(df_raw)
    df_clean = clip_outliers_iqr(df_filled, exclude_cols=[])

    clean_path = FIG_DIR / "新品销量预测数据集_预处理后.csv"
    df_clean.to_csv(clean_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 已保存: {clean_path}")

    (
        X_all,
        y_all,
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled_df,
        X_test_scaled_df,
        scaler,
    ) = split_and_scale(df_clean)

    plot_corr_heatmap(X_train)
    vif_df = compute_vif(X_train_scaled_df)

    ols_result = run_ols(X_train_scaled_df, X_test_scaled_df, y_train, y_test)

    lambdas_trace = np.linspace(0, 100, 200)
    plot_ridge_trace(
        X_train_scaled_df, y_train, X_train.columns.tolist(), lambdas_trace
    )

    lambdas_cv = np.logspace(-3, 2, 100)
    best_lambda, cv_df = cross_validate_ridge(
        X_train_scaled_df, y_train, lambdas_cv, n_splits=5
    )
    cv_df.to_csv(FIG_DIR / "交叉验证结果.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] 已保存: {FIG_DIR / '交叉验证结果.csv'}")
    plot_cv_curve(cv_df, best_lambda)

    ridge_result = run_ridge(
        X_train_scaled_df, X_test_scaled_df, y_train, y_test, best_lambda
    )

    save_tables(
        feature_names=X_train.columns.tolist(),
        true_beta=true_beta,
        vif_df=vif_df,
        ols_result=ols_result,
        ridge_result=ridge_result,
        best_lambda=best_lambda,
    )

    plot_true_vs_pred(
        y_test.values,
        ridge_result["y_test_pred"],
        "Ridge",
        "测试集真实值_vs_预测值散点图.png",
    )
    plot_residual_distribution(
        y_test.values, ridge_result["y_test_pred"], "Ridge", "测试集残差分布图.png"
    )
    plot_coef_bar(
        X_train.columns.tolist(),
        ridge_result["coef"],
        "Ridge 系数条形图",
        "Ridge系数条形图.png",
    )

    if ols_result["success"]:
        plot_coef_bar(
            X_train.columns.tolist(),
            ols_result["coef"],
            "OLS 系数条形图",
            "OLS系数条形图.png",
        )
    else:
        print("[WARN] OLS 未成功，跳过 OLS 系数条形图生成。")

    summary_lines = []
    summary_lines.append("岭回归实验运行完成")
    summary_lines.append(f"最优 lambda: {best_lambda:.6f}")
    if ols_result["success"]:
        summary_lines.append(f"OLS 测试集 R2: {ols_result['test_metrics']['R2']:.4f}")
    else:
        summary_lines.append(f"OLS 状态: 失败 ({ols_result['error']})")
    summary_lines.append(f"Ridge 测试集 R2: {ridge_result['test_metrics']['R2']:.4f}")
    summary_lines.append("已输出 CSV 表格与图片到 figures/ 目录")

    summary_path = FIG_DIR / "实验摘要.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"[OK] 已保存: {summary_path}")

    print("\n" + "=" * 80)
    print("[DONE] 实验运行完成，以下文件应已生成于 figures/ 目录：")
    print("1. 新品销量预测数据集.csv")
    print("2. 新品销量预测数据集_预处理后.csv")
    print("3. 特征相关性热力图.png")
    print("4. 岭迹图.png")
    print("5. 岭迹图系数轨迹.csv")
    print("6. 交叉验证结果.csv")
    print("7. 交叉验证MSE曲线.png")
    print("8. 性能对比表.csv")
    print("9. 系数对比表.csv")
    print("10. 特征重要性表.csv")
    print("11. VIF表.csv")
    print("12. 测试集真实值_vs_预测值散点图.png")
    print("13. 测试集残差分布图.png")
    print("14. Ridge系数条形图.png")
    print("15. OLS系数条形图.png（若OLS成功）")
    print("16. 实验摘要.txt")
    print("=" * 80)


if __name__ == "__main__":
    main()
