# Lab01 聚类实验任务书（Requirements）

来源：课程实验文档《Scikit-Learn 的学习与运用（聚类算法实践）》整理而成。  
目标：将原始实验要求转成可执行 checklist，便于复现与验收。

---ls -

## 1. 实验目标
- [ ] 理解无监督学习与聚类任务的基本概念（无标签、发现结构）
- [ ] 掌握并实现 KMeans 聚类（原理 + 参数 + 结果可视化）
- [ ] 学会选择 K：肘部法（Inertia）+ 轮廓系数（Silhouette）
- [ ] 学会对比：KMeans / 层次聚类（Agglomerative）/ DBSCAN
- [ ] Iris 数据集：用 PCA 降维到 2D 可视化并输出 ARI

---

## 2. 环境与依赖（必须可复现）
- [ ] 使用 conda 环境文件：`env/environment.yml`
- [ ] 关键依赖：scikit-learn / numpy / pandas / matplotlib / seaborn / scipy
- [ ] 能在环境中成功导入并打印版本（至少 sklearn）

验收：
- [ ] `python -c "import sklearn; print(sklearn.__version__)"` 输出版本号

---

## 3. 必做任务清单（按实验流程）
### 3.1 数据准备与预处理
- [ ] 使用 `make_blobs` 生成聚类数据集（用于演示聚类效果）
- [ ] 使用 `StandardScaler` 标准化特征（并解释原因）
- [ ] 使用 Iris 数据集（4维特征，3类），用于聚类对比

### 3.2 KMeans 聚类（核心任务）
- [ ] 训练 KMeans 并输出：labels、inertia、（必要时）cluster_centers
- [ ] 可视化聚类结果：
  - [ ] blobs（本身是 2D）直接画散点
  - [ ] Iris：先 PCA 到 2D 再画散点
- [ ] 在报告中解释 KMeans 原理：初始化质心 → 分配样本 → 更新质心 → 迭代至收敛
- [ ] 在报告中说明 KMeans 局限：初值敏感、凸簇偏好、异常值敏感
- [ ] 理解并说明参数：`random_state`、`n_init`、`max_iter`

### 3.3 选择 K（肘部法 + 轮廓系数）
- [ ] 画 K=2..10 的 Inertia 曲线（Elbow）
- [ ] 画 K=2..10 的 Silhouette 曲线
- [ ] 给出最终选择的 K，并解释依据（曲线 + 可视化）

### 3.4 层次聚类（Agglomerative）
- [ ] 运行 `AgglomerativeClustering`（常用 linkage=ward）
- [ ] 与 KMeans 对比（图 + silhouette）

### 3.5 DBSCAN（密度聚类）
- [ ] 运行 `DBSCAN` 并解释 `eps` / `min_samples`
- [ ] 正确处理噪声点：label = -1
- [ ] silhouette 计算时排除 -1 噪声点并说明原因

### 3.6 评估输出（指标）
- [ ] blobs：输出 silhouette（KMeans/层次/DBSCAN，DBSCAN 排除噪声）
- [ ] Iris：输出 silhouette，额外输出 ARI（与真实标签对比）
