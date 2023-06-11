import numpy as np
from sklearn.model_selection import train_test_split
from classifiers import Classifier
from preprocessing import get_fb_dataset, process_dataset
from config import settings
from matplotlib import pyplot as plt

# getting our arabic facebook dataset as pandas DataFrame
ds = get_fb_dataset(settings.dataset_path)

X, y = process_dataset(ds, test_ratio=0.1)

# splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=321)

# create a classifier class instance
clf = Classifier(X_train, X_test, y_train, y_test)

# plot out results
fig, axes = plt.subplots(nrows=1, ncols=3)
width = .25
max_iters_for_avg = 10

# params to test classifiers
n_trees = [3, 5, 7, 10, 20, 50]
max_depths = [10, 20, 50]
n_estimators = [5, 10, 15, 25, 50]
RF_scores = np.zeros((len(max_depths), len(n_trees)), dtype=np.float32)


for _ in range(max_iters_for_avg):
    seed = int(np.random.choice(100, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=seed)

    for i, d in enumerate(max_depths): 
        for j, n in enumerate(n_trees):
            RF_scores[i, j] += (1/max_iters_for_avg) * clf.apply_randomforest(n_trees=n, max_depth=d)

axes[0].bar(np.linspace(1,len(n_trees),len(n_trees))-width, RF_scores[0, :], width=width)
axes[0].bar(np.linspace(1,len(n_trees),len(n_trees)), RF_scores[1, :], width=width)
axes[0].bar(np.linspace(1,len(n_trees),len(n_trees))+width, RF_scores[2, :], width=width)
axes[0].set_xticks(ticks=np.linspace(1,len(n_trees),len(n_trees)), labels=n_trees)
'''
n_hiddens = [3, 5, 7, 9]
n_nodes = [5]
max_iters = [500, 1000, 5000]
ANN_scores = np.zeros((len(max_iters), len(n_hiddens)), dtype=np.float32)
for _ in range(max_iters_for_avg):
    seed = int(np.random.choice(100, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=seed)

    for i, t in enumerate(max_iters):
        for j, n in enumerate(n_hiddens):
            ANN_scores[i, j] += (1/max_iters_for_avg) * clf.apply_neuralnet(hidden_list=n*n_nodes, max_iters=t)

axes[1].bar(np.linspace(1,len(n_hiddens),len(n_hiddens))-width, ANN_scores[0, :], width=width)
axes[1].bar(np.linspace(1,len(n_hiddens),len(n_hiddens)), ANN_scores[1, :], width=width)
axes[1].bar(np.linspace(1,len(n_hiddens),len(n_hiddens))+width, ANN_scores[2, :], width=width)
axes[1].set_xticks(ticks=np.linspace(1,len(n_hiddens),len(n_hiddens)), labels=n_hiddens)
'''
gammas = [.1, .5, 1, 5, 10]
Cs = [.2, 1, 5]
max_iters = [500, 100, 5000]
svc_scores = np.zeros((len(Cs), len(gammas)), dtype=np.float32)
for _ in range(max_iters_for_avg):
    seed = int(np.random.choice(100, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=seed)

    for i, C in enumerate(Cs):
        for j, g in enumerate(gammas):
            svc_scores[i, j] += (1/max_iters_for_avg) * clf.apply_svc(gamma=g, c=C)

axes[2].bar(np.linspace(1,len(gammas),len(gammas))-width, svc_scores[0, :], width=width)
axes[2].bar(np.linspace(1,len(gammas),len(gammas)), svc_scores[1, :], width=width)
axes[2].bar(np.linspace(1,len(gammas),len(gammas))+width, svc_scores[2, :], width=width)
axes[2].set_xticks(ticks=np.linspace(1,len(gammas),len(gammas)), labels=gammas)

plt.show()
'''
samme_score = clf.apply_adaboost_sammer(n_estimators=5)
print('samme accuracy = ', samme_score)
'''