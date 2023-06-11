import numpy as np

class SVM():
    def __init__(self, lr=.001, lamda=.01, n_iters=1000):
        self.lr=lr
        self.n_iters=n_iters
        self.w=None
        self.b=None
        self.lamda=lamda

    def fit(self, X, y):
        _, n_features = X.shape   
        self.w, self.b = np.zeros(n_features), 0    
        y_ = np.where(y <= 0, -1, 1)

        for _ in range(self.n_iters):
            for idx, x in enumerate(X):
                cond = y_[idx] * (np.dot(x, self.w) - self.b) >= 1
                if not cond:
                    dw = 2*self.lamda*self.w - np.dot(x, y_[idx])
                    db = y_[idx]
                else:
                    dw = 2*self.lamda*self.w 
                self.w -= self.lr*dw
                self.b -= self.lr*db

    def predict(self, X):
        lin_output = np.dot(X, self.w) - self.b
        return np.sign(lin_output)
