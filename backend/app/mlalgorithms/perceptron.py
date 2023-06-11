import numpy as np

class Percepton():

    def __init__(self, lr=1e-3, iters=1e3) -> None:
        self.lr=lr
        self.iters=iters
        self.weights = None
        self.bias=None
        self.actFun=lambda x: np.where(x>0, 1, 0)

    # In fit we find the params
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # np.where function deals with vectors
        y_ = np.where(y<=0, 0, 1)
        for _ in range(self.iters):
            for i, x in enumerate(X):
                y_predicted = (np.dot(x, self.weights) + self.bias)
                delta = self.lr*(y_[i] - self.actFun(y_predicted))
                self.weights += delta*x
                self.bias += delta

    def predict(self, X):
        return self.actFun(np.dot(X, self.weights) + self.bias)