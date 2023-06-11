import numpy as np

class Stump:

    def __init__(self):
        self.polarity = 1
        self.feature = None
        self.thresh = None
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        Xc = X[:, self.feature]
        preds = np.ones(n_samples)
        if self.polarity == 1:
            preds[Xc < self.thresh] = -1
        else:
            preds[Xc > self.thresh] = -1
        return preds
        
class AdaBoost:

    def __init__(self, n_clfs):
        self.n_clfs = n_clfs

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.clfs = []
        w = np.full(n_samples, 1/n_samples)
         # Iterate through classifiers
        for _ in range(self.n_clfs):
            
            min_err = float("inf")
            clf = Stump()
            for f in range(n_features):
                Xc = X[:, f]
                for t in np.unique(Xc):
                    p = 1
                    preds = np.ones(n_samples)
                    preds[Xc < t] = -1
                    err = np.sum(w[y != preds])
                    if err > .5:
                        err = 1-err   
                        p = -1
                    if err < min_err:
                        min_err = err
                        clf.feature, clf.thresh, clf.polarity = f, t, p

            EPS = 1e-10
            clf.alpha = 0.5 * np.log((1.0-min_err+EPS)/(min_err + EPS))
            preds = clf.predict(X)
            w *= np.exp(-clf.alpha * y * preds)
            w/=sum(w)
            self.clfs.append(clf)

    def predict(self, X):
        return np.sign(np.sum([clf.alpha * clf.predict(X) for clf in self.clfs], axis=0))

                    