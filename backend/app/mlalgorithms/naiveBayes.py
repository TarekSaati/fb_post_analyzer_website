# argmax_y P(y|X) = argmax_y [P(x1|y).P(x2|y)...P(xn|y)] ~ argmax_y [log(P(x1|y)) + ... + log(P(xn|y))]

import numpy as np

class BayesClassifier():

    def fit(self, X, y):
        n_sasmples, n_features = X.shape   
        self._classes = np.unique(y)
        self.n_classes = len(self._classes)
        self.means = np.zeros((self.n_classes, n_features), dtype=np.float64)
        self.vars = np.zeros((self.n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(self.n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            Xc = X[c==y]
            self.means[idx,:] = Xc.mean(axis=0)
            self.vars[idx,:] = Xc.var(axis=0)
            self.priors[idx] = Xc.shape[0] / float(n_sasmples)

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        posteriors = []

        # enumerate returns the index and content
        for idx, _ in enumerate(self._classes):
            prior = np.log(self.priors[idx])
            class_cond = np.sum(np.log(self._pdf(idx, x)))
            posterior = class_cond+prior
            posteriors.append(posterior)
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_id, x):
        mean = self.means[class_id]
        var = self.vars[class_id]
        numerator = np.exp(-((x-mean)**2)/(2*var))
        denom = np.sqrt(2*np.pi*var)
        return numerator / denom