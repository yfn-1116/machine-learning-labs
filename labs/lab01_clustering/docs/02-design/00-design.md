# Lab01 方案设计（Design）

## 目标
- 将实验要求转为可复现的工程结构（run.py 一键生成图/指标）
- 输出目录固定：figures/ 与 outputs/
- 关键步骤：标准化、聚类、评估、可视化

## 目录约定
- src/：代码入口与模块
- docs/：需求/设计/FAQ/挑战/运行手册/知识库
- figures/：生成的图（默认不提交）
- outputs/：生成的指标与结果文件（默认不提交）

## 实现计划（TODO）
- [ ] 编写 src/run.py：生成 blobs 与 iris 的聚类图、inertia/silhouette 曲线、metrics.json
- [ ] 在 README 中说明复现命令与输出位置
