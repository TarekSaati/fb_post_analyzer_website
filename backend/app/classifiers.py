
from sklearn.metrics import accuracy_score
from mlalgorithms.random_forest import CustomRandomForest
from mlalgorithms.lda import LDA
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

class Classifier:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test

    def apply_svc(self, c=1.0, gamma=1):
        clf = SVC(kernel='rbf', gamma=gamma, C=c)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred)
    
    def apply_adaboost_sammer(self, n_estimators):
        clf = AdaBoostClassifier(n_estimators=n_estimators)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        # return accuracy_score(self.y_test, y_pred)
        return  clf.score(self.X_train, self.y_train), clf.score(self.X_test, self.y_test)

    def apply_customrandomforest(self, n_trees=7, min_samples_per_node=4, max_depth=10):
        clf = CustomRandomForest(n_trees=n_trees, min_samples_per_node=min_samples_per_node, max_depth=max_depth, n_feats=3)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred)
    
    def apply_randomforest(self, n_trees=7, min_samples_per_node=4, max_depth=10):
        clf = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth, min_samples_split=min_samples_per_node)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred)
    
    def apply_neuralnet(self, hidden_list, max_iters):
        clf = MLPClassifier(hidden_layer_sizes=hidden_list, max_iter=max_iters)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred)
    
    def apply_lda(self, n_components=2):
        lda = LDA(n_components)
        lda.fit(self.X_train, self.y_train)
        return lda.transform(self.X_train)
