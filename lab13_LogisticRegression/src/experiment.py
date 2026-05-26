import time
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.linear_model import LogisticRegression as SKLogisticRegression

from datasets import DATA_PATH, FIG_DIR, load_dataset, prepare_train_test
from logistic_regression import NativeLogisticRegression
from visualize import (
    plot_churn_distribution,
    plot_loss_curve,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_weights,
    plot_threshold_compare,
    plot_model_compare,
    plot_risk_distribution,
)


def evaluate_binary_model(model_name, y_true, y_pred, y_proba):
    return {
        "模型": model_name,
        "准确率": accuracy_score(y_true, y_pred),
        "精确率": precision_score(y_true, y_pred, zero_division=0),
        "召回率": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, y_proba),
    }


def build_risk_table(user_ids, proba):
    risk_df = pd.DataFrame({
        "用户ID": user_ids,
        "流失概率": proba,
    })

    risk_df["风险等级"] = pd.cut(
        risk_df["流失概率"],
        bins=[-0.01, 0.2, 0.4, 0.6, 0.8, 1.01],
        labels=["极低风险", "低风险", "中风险", "高风险", "极高风险"],
    )

    strategy_map = {
        "极高风险": "立即专属客服联系，大额优惠券或专属权益",
        "高风险": "定向推送个性化内容，中额优惠券",
        "中风险": "发送新功能通知，小额优惠券或任务激励",
        "低风险": "常规关怀，保持内容触达",
        "极低风险": "持续优化体验，避免过度打扰",
    }

    risk_df["建议策略"] = risk_df["风险等级"].astype(str).map(strategy_map)
    return risk_df.sort_values("流失概率", ascending=False)


def business_interpretation(weights_df, risk_df, metrics, save_path):
    top_risk = weights_df[weights_df["权重"] > 0].head(5)
    top_keep = weights_df[weights_df["权重"] < 0].head(5)

    risk_dist = risk_df["风险等级"].value_counts().sort_index()
    high_risk_count = int(risk_df["风险等级"].isin(["高风险", "极高风险"]).sum())

    estimated_save_rate = 0.20
    estimated_ltv = 500
    roi_value = high_risk_count * estimated_save_rate * estimated_ltv

    lines = []
    lines.append("数字运营用户流失预测业务解读")
    lines.append("=" * 60)
    lines.append("")
    lines.append("一、模型效果")
    lines.append(f"- 准确率: {metrics['准确率']:.4f}")
    lines.append(f"- 精确率: {metrics['精确率']:.4f}")
    lines.append(f"- 召回率: {metrics['召回率']:.4f}")
    lines.append(f"- F1: {metrics['F1']:.4f}")
    lines.append(f"- AUC: {metrics['AUC']:.4f}")
    lines.append("")
    lines.append("二、增加流失风险的关键特征")
    for _, row in top_risk.iterrows():
        lines.append(f"- {row['特征']}: 权重={row['权重']:.4f}，说明该指标升高会提高流失概率。")

    lines.append("")
    lines.append("三、降低流失风险的关键特征")
    for _, row in top_keep.iterrows():
        lines.append(f"- {row['特征']}: 权重={row['权重']:.4f}，说明该指标升高会降低流失概率。")

    lines.append("")
    lines.append("四、风险分层结果")
    for level, count in risk_dist.items():
        lines.append(f"- {level}: {count} 人")

    lines.append("")
    lines.append("五、ROI估算示例")
    lines.append(f"- 高风险与极高风险用户共 {high_risk_count} 人。")
    lines.append(f"- 假设挽留成功率为20%，单个留存用户LTV为500元。")
    lines.append(f"- 预计可创造价值约: {roi_value:.2f} 元。")

    lines.append("")
    lines.append("六、运营建议")
    lines.append("- 对极高风险用户：优先人工客服触达，处理投诉和近期不活跃问题。")
    lines.append("- 对高风险用户：定向发放优惠券，结合用户历史消费偏好做个性化推荐。")
    lines.append("- 对中风险用户：推送新功能、新活动和轻量化任务激励。")
    lines.append("- 对低风险用户：避免过度打扰，以常规内容触达和体验优化为主。")

    Path(save_path).write_text("\n".join(lines), encoding="utf-8")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(DATA_PATH)
    pack = prepare_train_test(df)

    X_train = pack["X_train"]
    X_test = pack["X_test"]
    y_train = pack["y_train"]
    y_test = pack["y_test"]
    uid_test = pack["uid_test"]
    feature_names = pack["feature_names"]

    print("=" * 70)
    print("机器学习实验13：数字运营用户流失预测 - 逻辑回归原生实现")
    print("=" * 70)
    print(f"数据位置: {DATA_PATH}")
    print(f"图片输出位置: {FIG_DIR}")
    print(f"训练集样本数: {len(y_train)}")
    print(f"测试集样本数: {len(y_test)}")
    print(f"特征数量: {X_train.shape[1]}")
    print(f"总体流失比例: {df['是否流失'].mean():.2%}")

    plot_churn_distribution(df["是否流失"].to_numpy(), FIG_DIR / "churn_distribution.png")

    print("\n开始训练原生逻辑回归模型...")
    native_model = NativeLogisticRegression(
        learning_rate=0.05,
        n_iterations=1000,
        batch_size=32,
        regularization="l2",
        lambda_=0.01,
        class_weight="balanced",
        random_state=42,
        verbose=True,
    )

    start = time.perf_counter()
    native_model.fit(X_train, y_train, feature_names=feature_names)
    native_train_time = time.perf_counter() - start

    start = time.perf_counter()
    native_proba = native_model.predict_proba(X_test)
    native_pred = native_model.predict(X_test, threshold=0.5)
    native_predict_time = time.perf_counter() - start

    train_pred = native_model.predict(X_train, threshold=0.5)
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, native_pred)

    native_metrics = evaluate_binary_model("原生逻辑回归", y_test, native_pred, native_proba)
    native_metrics["训练准确率"] = train_acc
    native_metrics["测试准确率"] = test_acc
    native_metrics["训练时间(s)"] = native_train_time
    native_metrics["预测时间(s)"] = native_predict_time
    native_metrics["训练迭代次数"] = native_model.n_iterations
    native_metrics["最终损失"] = native_model.loss_history[-1]

    print("\n原生逻辑回归评估结果:")
    for k, v in native_metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    print("\n详细分类报告:")
    report_text = classification_report(y_test, native_pred, target_names=["留存", "流失"], zero_division=0)
    print(report_text)

    cm = confusion_matrix(y_test, native_pred)
    print("\n混淆矩阵:")
    print(cm)

    weights_df = native_model.get_feature_weights()
    weights_df.to_csv(FIG_DIR / "feature_weights.csv", index=False, encoding="utf-8-sig")

    print("\n特征权重Top10:")
    print(weights_df.head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    plot_loss_curve(native_model.loss_history, FIG_DIR / "loss_history.png")
    plot_confusion_matrix(cm, FIG_DIR / "confusion_matrix.png")
    plot_roc_curve(y_test, native_proba, FIG_DIR / "roc_curve.png")
    plot_feature_weights(weights_df, FIG_DIR / "feature_weights.png", top_n=12)

    threshold_rows = []
    for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:
        pred = native_model.predict(X_test, threshold=threshold)
        threshold_rows.append({
            "阈值": threshold,
            "准确率": accuracy_score(y_test, pred),
            "精确率": precision_score(y_test, pred, zero_division=0),
            "召回率": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "预测流失人数": int(np.sum(pred == 1)),
        })

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(FIG_DIR / "threshold_metrics.csv", index=False, encoding="utf-8-sig")
    plot_threshold_compare(threshold_df, FIG_DIR / "threshold_compare.png")

    print("\n不同阈值指标对比:")
    print(threshold_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    risk_df = build_risk_table(uid_test, native_proba)
    risk_df.to_csv(FIG_DIR / "risk_segmentation.csv", index=False, encoding="utf-8-sig")
    risk_df.head(30).to_csv(FIG_DIR / "top30_high_risk_users.csv", index=False, encoding="utf-8-sig")
    plot_risk_distribution(risk_df, FIG_DIR / "risk_distribution.png")

    print("\n风险等级分布:")
    print(risk_df["风险等级"].value_counts().sort_index())

    print("\n开始训练sklearn逻辑回归用于对比验证...")
    sk_model = SKLogisticRegression(
        penalty="l2",
        C=100,
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    sk_model.fit(X_train, y_train)
    sk_pred = sk_model.predict(X_test)
    sk_proba = sk_model.predict_proba(X_test)[:, 1]
    sk_metrics = evaluate_binary_model("sklearn逻辑回归", y_test, sk_pred, sk_proba)

    compare_df = pd.DataFrame([
        {
            "模型": "原生逻辑回归",
            "准确率": native_metrics["准确率"],
            "精确率": native_metrics["精确率"],
            "召回率": native_metrics["召回率"],
            "F1": native_metrics["F1"],
            "AUC": native_metrics["AUC"],
        },
        sk_metrics,
    ])
    compare_df.to_csv(FIG_DIR / "model_compare_metrics.csv", index=False, encoding="utf-8-sig")
    plot_model_compare(compare_df, FIG_DIR / "model_compare.png")

    sk_weights = pd.DataFrame({
        "特征": feature_names,
        "sklearn权重": sk_model.coef_[0],
    })

    weight_compare = pd.merge(
        weights_df[["特征", "权重", "影响方向"]],
        sk_weights,
        on="特征",
        how="left",
    )
    weight_compare.to_csv(FIG_DIR / "weight_compare_with_sklearn.csv", index=False, encoding="utf-8-sig")

    print("\n原生实现与sklearn对比:")
    print(compare_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    with open(FIG_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write("原生逻辑回归详细分类报告\n")
        f.write(report_text)
        f.write("\n\n不同阈值指标对比\n")
        f.write(threshold_df.to_string(index=False))
        f.write("\n\n原生实现与sklearn对比\n")
        f.write(compare_df.to_string(index=False))

    result_metrics = pd.DataFrame([native_metrics])
    result_metrics.to_csv(FIG_DIR / "metrics_summary.csv", index=False, encoding="utf-8-sig")

    business_interpretation(
        weights_df,
        risk_df,
        native_metrics,
        FIG_DIR / "business_interpretation.txt",
    )

    print("\n已生成图片和结果文件：")
    for path in sorted(Path(FIG_DIR).glob("*")):
        print(" -", path)


if __name__ == "__main__":
    main()
