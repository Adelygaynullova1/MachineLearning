import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
if __name__ == '__main__':
    iris = load_iris()
    X = iris.data

    inertia_values = []

    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertia_values.append(kmeans.inertia_)

    plt.plot(range(1, 11), inertia_values, marker='o')
    plt.xlabel('Номер кластера')
    plt.ylabel('Инерция')
    plt.show()


    diffs = np.diff(inertia_values)


    threshold = 0.1


    optimal_k_index = np.where(diffs < threshold)[0][0] + 2

    optimal_k = optimal_k_index + 1

    print(optimal_k)