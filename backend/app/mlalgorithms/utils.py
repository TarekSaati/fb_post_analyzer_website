import numpy as np

def euc_dist(x, y):
    return np.sqrt(np.sum((x-y)**2))

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy
