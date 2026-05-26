import time
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from datasets import (
    CLASS_NAMES,
    DATA_PATH,
    FIG_DIR,
    get_train_test,
    load_dataset,
    preprocess_dataset,
)
from gbdt import GBDTMultiClass
from visualize import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_hyperparam_compare,
    plot_loss_curve,
    plot_model_compare,
    plot_overfitting_compare,
    plot_pr_curve_compare,
)


def train_with_time(model, X_train, y_train):
    start = time.perf_counter()
    model.fit(X_train, y_train)
    return model, time.perf_counter() - start


def predict_with_time(model, X):
    start = time.perf_counter()
    y_pred = model.predict(X)
    return y_pred, time.perf_counter() - start


def evaluate_model(name, model, X_train, X_test, y_train, y_test, train_time, pred_time):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    weighted = precision_recall_fscore_support(
        y_test, y_test_pred, average="weighted", zero_division=0
    )

    burst_recall = precision_recall_fscore_support(
        y_test, y_test_pred, labels=[3], average=None, zero_division=0
    )[1][0]

    return {
        "模型": name,
        "核心参数": getattr(model, "param_desc", ""),
        "训练准确率": accuracy_score(y_train, y_train_pred),
        "测试准确率": accuracy_score(y_test, y_test_pred),
        "加权精确率": weighted[0],
        "加权召回率": weighted[1],
        "加权F1": weighted[2],
        "爆款召回率": burst_recall,
        "训练时间(s)": train_time,
        "预测时间(s)": pred_time,
    }


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(DATA_PATH)
    X, y, feature_names, raw_df = preprocess_dataset(df)
    X_train, X_test, y_train, y_test = get_train_test(X, y)

    print("=" * 70)
    print("机器学习实验12：电商直播流量转化率预测 - GBDT原生实现")
    print("=" * 70)
    print(f"数据位置: {DATA_PATH}")
    print(f"图片输出位置: {FIG_DIR}")
    print(f"样本数: {len(y)}，特征数: {X.shape[1]}")
    print("标签分布:", pd.Series(y).value_counts().sort_index().to_dict())

    plot_class_distribution(y, CLASS_NAMES, FIG_DIR / "class_distribution.png")

    gbdt = GBDTMultiClass(
        n_estimators=20,
        learning_rate=0.20,
        max_depth=3,
        min_samples_split=6,
    )
    gbdt.param_desc = "n_estimators=20,max_depth=3,lr=0.20"

    print("\n开始训练原生GBDT...")
    gbdt, gbdt_train_time = train_with_time(gbdt, X_train, y_train)
    gbdt_pred, gbdt_pred_time = predict_with_time(gbdt, X_test)

    rf = RandomForestClassifier(
        n_estimators=20,
        max_depth=4,
        random_state=42,
        class_weight="balanced",
    )
    rf.param_desc = "n_estimators=20,max_depth=4"

    print("开始训练随机森林对比模型...")
    rf, rf_train_time = train_with_time(rf, X_train, y_train)
    rf_pred, rf_pred_time = predict_with_time(rf, X_test)

    rows = [
        evaluate_model("原生GBDT", gbdt, X_train, X_test, y_train, y_test, gbdt_train_time, gbdt_pred_time),
        evaluate_model("随机森林", rf, X_train, X_test, y_train, y_test, rf_train_time, rf_pred_time),
    ]

    results_df = pd.DataFrame(rows)
    results_df.to_csv(FIG_DIR / "metrics_compare.csv", index=False, encoding="utf-8-sig")

    print("\n模型指标对比:")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n原生GBDT详细分类报告:")
    gbdt_report = classification_report(y_test, gbdt_pred, target_names=CLASS_NAMES, zero_division=0)
    print(gbdt_report)

    print("\n随机森林详细分类报告:")
    rf_report = classification_report(y_test, rf_pred, target_names=CLASS_NAMES, zero_division=0)
    print(rf_report)

    with open(FIG_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write("原生GBDT详细分类报告\n")
        f.write(gbdt_report)
        f.write("\n随机森林详细分类报告\n")
        f.write(rf_report)

    plot_confusion_matrix(
        confusion_matrix(y_test, gbdt_pred),
        CLASS_NAMES,
        "原生GBDT混淆矩阵",
        FIG_DIR / "gbdt_confusion_matrix.png",
    )

    plot_confusion_matrix(
        confusion_matrix(y_test, rf_pred),
        CLASS_NAMES,
        "随机森林混淆矩阵",
        FIG_DIR / "rf_confusion_matrix.png",
    )

    plot_feature_importance(
        gbdt.feature_importances_,
        feature_names,
        FIG_DIR / "gbdt_feature_importance.png",
    )

    plot_model_compare(results_df, FIG_DIR / "model_compare.png")
    plot_overfitting_compare(results_df, FIG_DIR / "overfitting_compare.png")
    plot_loss_curve(gbdt.loss_history_, FIG_DIR / "gbdt_loss_curve.png")

    proba_dict = {
        "原生GBDT": gbdt.predict_proba(X_test),
        "随机森林": rf.predict_proba(X_test),
    }
    plot_pr_curve_compare(
        y_test,
        proba_dict,
        3,
        "爆款转化",
        FIG_DIR / "burst_pr_curve_compare.png",
    )

    hyper_rows = []
    hyper_params = [
        (10, 0.10, 2),
        (20, 0.10, 3),
        (20, 0.20, 3),
        (30, 0.20, 3),
        (20, 0.20, 4),
    ]

    print("\n开始GBDT超参数对比实验...")
    for n_estimators, lr, depth in hyper_params:
        model = GBDTMultiClass(
            n_estimators=n_estimators,
            learning_rate=lr,
            max_depth=depth,
            min_samples_split=6,
        )

        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - start

        pred = model.predict(X_test)
        burst_recall = precision_recall_fscore_support(
            y_test, pred, labels=[3], average=None, zero_division=0
        )[1][0]

        hyper_rows.append({
            "参数组合": f"树{n_estimators}_lr{lr}_深度{depth}",
            "n_estimators": n_estimators,
            "learning_rate": lr,
            "max_depth": depth,
            "测试准确率": accuracy_score(y_test, pred),
            "爆款召回率": burst_recall,
            "训练时间(s)": train_time,
        })

    hyper_df = pd.DataFrame(hyper_rows)
    hyper_df.to_csv(FIG_DIR / "gbdt_hyperparam_results.csv", index=False, encoding="utf-8-sig")
    plot_hyperparam_compare(hyper_df, FIG_DIR / "gbdt_hyperparam_compare.png")

    importance_df = pd.DataFrame({
        "特征": feature_names,
        "重要性": gbdt.feature_importances_,
    }).sort_values("重要性", ascending=False)

    importance_df.to_csv(FIG_DIR / "gbdt_feature_importance.csv", index=False, encoding="utf-8-sig")

    print("\n特征重要性Top5:")
    print(importance_df.head(5).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n已生成图片和结果文件：")
    for path in sorted(Path(FIG_DIR).glob("*")):
        print(" -", path)


if __name__ == "__main__":
    main()
