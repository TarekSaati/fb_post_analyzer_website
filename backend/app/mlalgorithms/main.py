from utils import accuracy
from lda import LDA
from adaboost import AdaBoost
from kmeans import KMeans
from pca import PCA
from decision_tree import DecisionTree
from matplotlib.colors import ListedColormap
import numpy as np
from knn import KNN
from linearreg import LinReg, LogReg
from naiveBayes import BayesClassifier
from sklearn import datasets
from perceptron import Percepton
from random_forest import RandomForest
from svm import SVM
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)
predictor = KNN(X_train, y_train, 7)
Yp = predictor.predict(X_test)
print(accuracy(y_test, Yp))
'''
def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

X, y = datasets.make_regression(
    n_samples=100, n_features=1, noise=20, random_state=4
)

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

reg = LinReg(lr=1e-2, n_iter=1000)
reg.fit(X_train, y_train)
res = reg.predict(X_test)
print(mean_squared_error(y_test, res))

y_pred_line = reg.predict(X)
cmap = plt.get_cmap("viridis")
# fig = plt.figure(figsize=(8, 6))
# m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
# m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
# plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
# plt.show()

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

regressor = LogReg(lr=0.001, n_iter=1000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

print("LR classification accuracy:", accuracy(y_test, predictions))

per = Percepton(lr=.1, iters=1000)
per.fit(X_train, y_train)
predictions = per.predict(X_test)
print("Perceptron classification accuracy:", accuracy(y_test, predictions))

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

# x0_1 = np.amin(X_train[:, 0])
# x0_2 = np.amax(X_train[:, 0])

# x1_1 = (-per.weights[0] * x0_1 - per.bias) / per.weights[1]
# x1_2 = (-per.weights[0] * x0_2 - per.bias) / per.weights[1]

# ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

# ymin = np.amin(X_train[:, 1])
# ymax = np.amax(X_train[:, 1])
# ax.set_ylim([ymin - 3, ymax + 3])

# plt.show()
'''
X, y = datasets.make_classification(
        n_samples=1000, n_features=10, n_classes=2, random_state=123
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

nb = BayesClassifier()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)
print("Naive Bayes classification accuracy", accuracy(y_test, predictions))
'''
#======================================================
X, y = datasets.make_blobs(
        n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=30
    )
y = np.where(y == 0, -1, 1)
clf = SVM(lamda=0)
clf.fit(X, y)

print(clf.w, clf.b)

def visualize_svm():
    def get_hyperplane_value(x, w, b, offset):
        return (-w[0] * x + b + offset) / w[1]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)

    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
    x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)

    x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)
    x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)

    x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)
    x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

    x1_min = np.amin(X[:, 1])
    x1_max = np.amax(X[:, 1])
    ax.set_ylim([x1_min - 3, x1_max + 3])

    plt.show()

visualize_svm()

#=============================Decision tree================
data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

clf = DecisionTree(max_depth=10)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy(y_test, y_pred)

print("Single DT Accuracy:", acc)

clf = RandomForest(n_trees=10, max_depth=10)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy(y_test, y_pred)

print("RT Accuracy:", acc)

#=========================== PCA ==============================
# data = datasets.load_digits()
data = datasets.load_iris()
X = data.data
y = data.target

# Project the data onto the 2 primary principal components
pca = PCA(3)
pca.fit(X)
X_projected = pca.transform(X)

print("Shape of X:", X.shape)
print("Shape of transformed X:", X_projected.shape)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

plt.scatter(
    x2, x2, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3)
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
plt.show()

#========================= K MEANS ====================================
from sklearn.datasets import make_blobs

X, y = make_blobs(
    centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40
)
print(X.shape)

clusters = len(np.unique(y))
print(clusters)

k = KMeans(X, k=clusters, max_iters=150, plot_steps=True)
y_pred = k.predict()
print('centroids: ', y_pred)

k.plot()


#============================ AdaBoost ==============================
data = datasets.load_breast_cancer()
X, y = data.data, data.target

y[y == 0] = -1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5
)

# Adaboost classification with 5 weak classifiers
clf = AdaBoost(n_clfs=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy(y_test, y_pred)
print("Accuracy:", acc)

# ============================= LDA ===============================
data = datasets.load_iris()
X, y = data.data, data.target

# Project the data onto the 2 primary linear discriminants
lda = LDA(2)
lda.fit(X, y)
X_projected = lda.transform(X)

print("Shape of X:", X.shape)
print("Shape of transformed X:", X_projected.shape)

x1, x2 = X_projected[:, 0], X_projected[:, 1]

plt.scatter(
    x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3)
)

plt.xlabel("Linear Discriminant 1")
plt.ylabel("Linear Discriminant 2")
plt.colorbar()
plt.show()
'''