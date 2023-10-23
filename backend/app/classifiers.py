
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# RF
n_trees = [3, 5, 7, 10, 20, 50]
max_depths = [10, 20, 50]
n_estimators = [5, 10, 15, 25, 50]
# ANNs
n_hiddens = [3, 5, 7, 9]
n_nodes = [5]
max_iters = [500, 1000, 5000]
#SVC
gammas = [.1, .5, 1, 5, 10]
Cs = [.2, 1, 5]
max_iters = [500, 100, 5000]

class Classifier:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test    
        self.clf = None

    def eval(self):
        raise NotImplementedError()
    
    def predict(self):
        raise NotImplementedError()

class MySVC(Classifier):

    def __init__(self, X_train, X_test, y_train, y_test, *args, isOptimal=False):
        super().__init__(X_train, X_test, y_train, y_test)
        self._optC = 1.0
        self.clf = SVC(kernel='rbf',
                    gamma='scale' if isOptimal else args[0],
                    C=self._optC if isOptimal else args[1]).fit(X_train, y_train)
    
    def eval(self):
        y_pred = self.clf.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred)
    def predict(self, X):
        return self.clf.predict(X)
    
class MyRF(Classifier):

    def __init__(self, X_train, X_test, y_train, y_test, *args, isOptimal=False):
        super().__init__(X_train, X_test, y_train, y_test)
        self.optNtrees = 7
        self.optDepth = 10
        self.clf = RandomForestClassifier(n_estimators=self.optNtrees if isOptimal else args[0],
        max_depth=self.optDepth if isOptimal else args[1],
        min_samples_split=4).fit(X_train, y_train)

    def eval(self):
        y_pred = self.clf.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred)
    def predict(self, X):
        return self.clf.predict(X)
    
class MyADABOOST(Classifier):

    def __init__(self, X_train, X_test, y_train, y_test, *args, isOptimal=False):
        super().__init__(X_train, X_test, y_train, y_test)
        self.optNestimators = 10
        self.clf = AdaBoostClassifier(n_estimators=self.optNestimators if isOptimal else args[0]).fit(X_train, y_train)
    
    def eval(self):
        y_pred = self.clf.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred)
    
    def predict(self, X):
        return self.clf.predict(X)
    
class MyANN(Classifier):

    def __init__(self, X_train, X_test, y_train, y_test, *args, isOptimal=False):
        super().__init__(X_train, X_test, y_train, y_test)
        self.optHidden = [5] * 7
        self.optIters = 1000
        self.clf = MLPClassifier(hidden_layer_sizes=self.optHidden if isOptimal else ([5] * args[0]),
        max_iter=self.optIters if isOptimal else args[1]).fit(X_train, y_train)

    def eval(self):
        y_pred = self.clf.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred)
    
    def predict(self, X):
        return self.clf.predict(X)
