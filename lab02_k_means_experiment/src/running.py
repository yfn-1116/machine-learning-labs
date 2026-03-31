# 导入必要的内置模块
import random
import math
import matplotlib.pyplot as plt

# ===================== 模块1：生成人工测试数据集 =====================
def generate_data():
    """
    生成人工构造的二维数据集：3个簇，每个簇10个样本（带轻微噪声）
    簇1中心：(2,3)，簇2中心：(8,7)，簇3中心：(5,12)
    返回：样本列表，每个元素为(x,y)元组
    """
    data = []

    for _ in range(10):
        data.append((2 + random.uniform(-1,1), 3 + random.uniform(-1,1)))
    for _ in range(10):
        data.append((8 + random.uniform(-1,1), 7 + random.uniform(-1,1)))
    for _ in range(10):
        data.append((5 + random.uniform(-1,1), 12 + random.uniform(-1,1)))

    return data

# ===================== 模块2：计算欧氏距离 =====================
def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# ===================== 模块3：K-means++ 初始化 =====================
def init_centroids(dataset, k):
    """
    使用 K-means++ 方法初始化质心
    """
    centroids = [random.choice(dataset)]

    while len(centroids) < k:
        candidate_points = []
        candidate_weights = []

        for point in dataset:
            if point in centroids:
                continue
            min_dist = min(euclidean_distance(point, c) for c in centroids)
            candidate_points.append(point)
            candidate_weights.append(min_dist**2)

        if not candidate_points:
            break

        if sum(candidate_weights) == 0:
            centroids.append(random.choice(candidate_points))
            continue

        next_centroid = random.choices(candidate_points, weights=candidate_weights, k=1)[0]
        centroids.append(next_centroid)

    return centroids

# ===================== 模块4：分配样本到对应的簇 =====================
def assign_clusters(dataset, centroids):
    k = len(centroids)
    clusters = {i: [] for i in range(k)}

    for point in dataset:
        distances = [euclidean_distance(point, c) for c in centroids]
        idx = distances.index(min(distances))
        clusters[idx].append(point)

    return clusters

# ===================== 模块5：更新各簇质心 =====================
def update_centroids(clusters, dataset):
    new_centroids = []
    for idx in clusters:
        points = clusters[idx]
        if not points:
            new_centroids.append(random.choice(dataset))
        else:
            mean_x = sum(p[0] for p in points)/len(points)
            mean_y = sum(p[1] for p in points)/len(points)
            new_centroids.append((mean_x, mean_y))
    return new_centroids

# ===================== 模块6：K-means 主函数 =====================
def k_means(dataset, k, max_iter=100, tol=1e-4):
    centroids = init_centroids(dataset, k)
    iter_count = 0

    while iter_count < max_iter:
        clusters = assign_clusters(dataset, centroids)
        new_centroids = update_centroids(clusters, dataset)

        centroid_change = sum(euclidean_distance(c1,c2) for c1,c2 in zip(centroids,new_centroids))

        if centroid_change < tol:
            centroids = new_centroids
            clusters = assign_clusters(dataset, centroids) # 保证最终簇对应质心
            print(f"迭代{iter_count+1}次后收敛，质心变化量：{centroid_change:.6f}")
            break

        centroids = new_centroids
        iter_count += 1

    if iter_count == max_iter:
        print(f"达到最大迭代次数 {max_iter}，停止迭代")

    return clusters, centroids

# ===================== 模块7：可视化 =====================
def plot_clusters(clusters, centroids, filename="kmeans_result.png"):
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

    plt.figure(figsize=(8,6))
    for idx in clusters:
        pts = clusters[idx]
        if pts:
            x = [p[0] for p in pts]
            y = [p[1] for p in pts]
            plt.scatter(x, y, label=f'Cluster {idx+1}', color=colors[idx%len(colors)])

    cx = [c[0] for c in centroids]
    cy = [c[1] for c in centroids]
    plt.scatter(cx, cy, marker='*', s=250, color='black', label='Centroids')

    for i,(c_x,c_y) in enumerate(centroids):
        plt.text(c_x+0.1, c_y+0.1, f'C{i+1}', fontsize=10)

    plt.title('K-means++ Clustering Result')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # 保存图像
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    # 显示图像
    plt.show(block=True)
    plt.close()

# ===================== 主程序入口 =====================
if __name__ == "__main__":
    dataset = generate_data()
    print(f"测试数据集生成完成，共 {len(dataset)} 个样本")
    print(f"前3个样本：{dataset[:3]}\n")

    K = 3
    MAX_ITER = 100
    TOL = 1e-6

    # 运行 K-means++ 算法
    final_clusters, final_centroids = k_means(dataset, K, MAX_ITER, TOL)

    print("===== 最终聚类结果 =====")
    for idx in final_clusters:
        print(f"\n簇{idx+1}：")
        print(f"  - 样本数量：{len(final_clusters[idx])}")
        print(f"  - 质心坐标：({final_centroids[idx][0]:.2f}, {final_centroids[idx][1]:.2f})")
        print(f"  - 前3个样本：{[(round(p[0],2),round(p[1],2)) for p in final_clusters[idx][:3]]}")

    # 可视化
    plot_clusters(final_clusters, final_centroids)