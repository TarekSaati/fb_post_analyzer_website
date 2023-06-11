import numpy as np
from matplotlib import pyplot as plt

from utils import euc_dist

np.random.seed(22)

class KMeans:
    def __init__(self, X, k=2, max_iters=150, plot_steps=True):
        self.k = k
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        self.centroids = []
        self.clusters = []
        self.X = X

    
    def init_centeriods(self):
        n_features = self.X.shape[1]
        max_bounds, min_bounds = [np.max(self.X[:, i]) for i in range(n_features)], [np.min(self.X[:, i]) for i in range(n_features)]
        for j in range(self.k):
            self.centroids.append([np.random.uniform(min_bounds[i], max_bounds[i]) for i in range(n_features)])
    
    def _euclidean_distances_to_centroids(self):
        n_samples = self.X.shape[0]
        dists = []
        for j in range(self.k):
            point_dist = [euc_dist(self.X[i], np.array(self.centroids[j])) for i in range(n_samples)]
            dists.append(point_dist) 
        return dists
    
    def _assign_clusters(self):
        dists = self._euclidean_distances_to_centroids()
        labels = np.argmin(dists, axis=0)
        # list of nd arrays
        self.clusters = [(self.X[labels == i]) for i in range(self.k)]
        return np.array(labels)
    
    def _update_centroids(self):
        n_features = self.X.shape[1]  
        self.centroids = [[sum(self.clusters[j][:, k])/len(self.clusters[j]) for k in range(n_features)] for j in range(self.k)]
        
    def _is_converged(self, centroids_old):
        # distances between each old and new centroids, fol all centroids
        distances = [
            euc_dist(np.array(centroids_old[i]), np.array(self.centroids[i])) for i in range(self.k)
        ]
        return sum(distances) == 0

    def predict(self):
        self.init_centeriods()
        if self.plot_steps:
                self.plot()
        for _ in range(self.max_iters):                 
            old_centroids = self.centroids
            self._assign_clusters()
            self._update_centroids()
            if self._is_converged(old_centroids):
                break
            if self.plot_steps:
                self.plot()
        return self.centroids
    
    def plot(self):
        _, ax = plt.subplots(figsize=(12, 8))

        for idx, cluster in enumerate(self.clusters):
            point = self.clusters[idx].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()
