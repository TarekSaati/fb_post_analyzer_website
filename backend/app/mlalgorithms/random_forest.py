from collections import Counter
import numpy as np
from mlalgorithms.decision_tree import DecisionTree

class CustomRandomForest:

    def __init__(self, n_trees=4, max_depth=100, min_samples_per_node=2, n_feats=None):
        self.n_trees = n_trees
        self.max_depth=max_depth
        self.min_samples_by_node=min_samples_per_node
        self.n_feats = n_feats
        self.trees = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(self.min_samples_by_node, self.max_depth, self.n_feats)
            # randomIds = np.random.choice(n_samples, n_samples, replace=False).flatten()
            # tree.fit(X[randomIds], y[randomIds])
            randomIds = np.random.choice(n_samples, n_samples, replace=True).flatten()
            tree.fit(X[randomIds], y[randomIds])
            self.trees.append(tree)

    def predict(self, X):
        n_test = X.shape[0]
        votes = [tree.predict(X) for tree in self.trees]
        votes = np.swapaxes(votes, 0, 1)
        result = [Counter(votes[v]).most_common(1)[0][0] for v in range(n_test)]
        return np.array(result)
            


