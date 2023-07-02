from collections import Counter
import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # need mean
        self.mean = np.mean(X, axis=0)
        # calc auto-covarience with np.cov feats*samples
        cov = np.cov((X - self.mean).T)
        # calc eignvectors and eigenvalues
        eignvals, eignvects = np.linalg.eig(cov)
        ids = np.argsort(eignvals, axis=0)[::-1]
        ids = ids[:self.n_components]
        self.components = eignvects[ids]

    def transform(self, X):
        Xm = X - self.mean
        return np.dot(Xm, self.components.T)