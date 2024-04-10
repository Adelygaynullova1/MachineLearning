import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
if __name__ == '__main__':

    plt.ion()

    iris = load_iris()
    X = iris.data

    def dist(p1, p2):
        return np.sqrt(np.sum((p1 - p2)**2))

    def set_centroids(X, k):
        indices = np.random.choice(X.shape[0], k, replace=False)
        return X[indices]

    def assign_clusters(X, centroids):
        clusters = []
        for point in X:
            distances = [dist(point, centroid) for centroid in centroids]
            cluster = np.argmin(distances)
            clusters.append(cluster)
        return np.array(clusters)

    def new_centroids(X, clusters, k):
        centroids = np.zeros((k, X.shape[1]))
        for cluster in range(k):
            cluster_points = X[clusters == cluster]
            centroids[cluster] = np.mean(cluster_points, axis=0)
        return centroids

    def visualize_clusters(X, centroids, clusters, step):
        plt.figure(figsize=(8, 6))
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink']
        colors_to_use = colors[:len(centroids)]
        for cluster in range(len(centroids)):
            cluster_points = X[clusters == cluster]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors_to_use[cluster], label=f'Cluster {cluster}')
            plt.scatter(centroids[cluster, 0], centroids[cluster, 1], c='black', marker='x', s=100)
        plt.title(f'Шаг {step}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.pause(1)


    def k_means(X, k, max_iters=100):
        centroids = set_centroids(X, k)
        step = 0
        while True:
            clusters = assign_clusters(X, centroids)
            visualize_clusters(X, centroids, clusters, step)
            n_centroids = new_centroids(X, clusters, k)
            if np.all(centroids == n_centroids) or step >= max_iters:
                break
            centroids = n_centroids
            step += 1
        plt.ioff()
        plt.show()

    k_means(X[:, :2], 3)