import numpy as np

class LDA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.linear_discriminants = None

    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)
        mean_overall = np.mean(X, axis=0)

        # calc S_Within: S_W = sum_over_classes((x-mean)*(x-mean)) 
        # S_B = sum_over_classes( n_c * (mean_X_c - mean_overall)^2 )
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))
        for c in class_labels:
            Xc = X[y==c]
            mean_c = np.mean(Xc, axis=0)
            S_W += (Xc-mean_c).T.dot(Xc-mean_c)
            n_c = Xc.shape[0]
            S_B += n_c*(mean_c-mean_overall).T.dot(mean_c-mean_overall)

        A = np.linalg.inv(S_W).dot(S_B)
        eigvals, eigvects = np.linalg.eig(A)
        ids = np.argsort(eigvals, axis=0)[::-1]
        ids = ids[:self.n_components]
        self.linear_discriminants = eigvects[0 : self.n_components]
        
    def transform(self, X):
        # project data
        return np.dot(X, self.linear_discriminants.T)
