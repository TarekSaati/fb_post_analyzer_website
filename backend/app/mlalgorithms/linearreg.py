import numpy as np

class Regressor():
    def __init__(self, lr=1e-2, n_iter=1000):
        self.lr=lr
        self.n_iter=n_iter
        self.w=None
        self.b=None

    def fit(self, X, y):
        n_sasmples, n_features = X.shape   
        self.w, self.b = np.zeros(n_features), 0    
        for _ in range(self.n_iter):
            y_pred = self._approximate(X)
            dw = 2*(1/n_sasmples)*(np.dot(X.T, y_pred - y))
            db = 2*(1/n_sasmples)*np.sum(y_pred - y)
            self.w -= self.lr*dw
            self.b -= self.lr*db

    def _approximate(self, X, w, b):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()

class LinReg(Regressor):

    def _approximate(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        return np.dot(X,  self.w) + self.b
    
class LogReg(Regressor):

    def _approximate(self, X):
        linApprox = np.dot(X, self.w) + self.b
        return self._segmoid(linApprox)
    
    def _segmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, X):
        return [1 if self._approximate(x) >= 0.5 else 0 for x in X]