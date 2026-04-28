import numpy as np


class NativeDBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        self.core_sample_indices_ = None

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _find_eps_neighborhood(self, X, p_idx):
        neighbors = []
        for q_idx in range(X.shape[0]):
            if self._euclidean_distance(X[p_idx], X[q_idx]) <= self.eps:
                neighbors.append(q_idx)
        return neighbors

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]

        labels = np.full(n_samples, -2, dtype=int)  # -2 未访问
        core_indices = []
        cluster_id = 0

        for p_idx in range(n_samples):
            if labels[p_idx] != -2:
                continue

            neighbors = self._find_eps_neighborhood(X, p_idx)

            if len(neighbors) < self.min_samples:
                labels[p_idx] = -1  # 噪声
                continue

            labels[p_idx] = cluster_id
            if p_idx not in core_indices:
                core_indices.append(p_idx)

            queue = neighbors.copy()

            while queue:
                q_idx = queue.pop(0)

                if labels[q_idx] == -2:
                    q_neighbors = self._find_eps_neighborhood(X, q_idx)

                    if len(q_neighbors) >= self.min_samples:
                        if q_idx not in core_indices:
                            core_indices.append(q_idx)

                        for idx in q_neighbors:
                            if idx not in queue:
                                queue.append(idx)

                if labels[q_idx] < 0:
                    labels[q_idx] = cluster_id

            cluster_id += 1

        self.labels_ = labels
        self.core_sample_indices_ = np.array(core_indices, dtype=int)
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
