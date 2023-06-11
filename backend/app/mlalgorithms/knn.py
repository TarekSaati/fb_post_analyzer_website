import utils
import numpy as np
from collections import Counter

class KNN:

    def __init__(self, X, y, k=3):
        self.k=k
        self.X=X
        self.y=y

    def predict(self, Xp):
        return [self._predict(x) for x in Xp]
           
    
    def _predict(self, xp):
        ds = [utils.euc_dist(X, xp) for X in self.X]
        knearest = np.argsort(ds)[:self.k]
        return Counter([self.y[ind] for ind in knearest]).most_common(1)[0][0]
    