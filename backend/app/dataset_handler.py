import numpy as np
from sklearn.model_selection import train_test_split
from .classifiers import MyADABOOST, MyANN, MyRF, MySVC
from .preprocessing import process_dataset
from .config import settings
from matplotlib import pyplot as plt
from sqlalchemy import create_engine
import pandas as pd

def prepare_classifier(clfName: str, seed: int):
    # importing the arabic facebook dataset from database as pandas DataFrame
    DATABASE_URL = f'postgresql://{settings.db_username}:{settings.db_password}@{settings.db_host}/{settings.db_name}'
    engine = create_engine(DATABASE_URL)
    df = pd.read_sql_table('posts',
                            engine,
                            columns=[
                                'likes', 'comments', 'shares',
                                'value', 'time', 'timestamp',
                                'topic', 'pagename'
                                ],
                            index_col='index')

    X, y = process_dataset(df, test_ratio=0.1)

    # train-test splitting 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=seed)

    # create a classifier class instance
    if clfName == 'svc':
        opt = MySVC(X_train, X_test, y_train, y_test, isOptimal=True)
    elif clfName == 'rf':
        opt = MyRF(X_train, X_test, y_train, y_test, isOptimal=True)
    elif clfName == 'ann':
        opt = MyANN(X_train, X_test, y_train, y_test, isOptimal=True)
    else:
        opt = MyADABOOST(X_train, X_test, y_train, y_test, isOptimal=True)
    
    return opt.clf

def plot_clf_stats(clfName, **kwargs):
    # plot out results
    width = .25
    max_iters_for_avg = 10
    params = kwargs.values()

    scores = np.zeros((len(params[1]), len(params[0])), dtype=np.float32)

    for _ in range(max_iters_for_avg):
        seed = int(np.random.choice(100, 1))
        clf = prepare_classifier(clfName=clfName, seed=seed)

        for i, d in enumerate(params[1]): 
            for j, n in enumerate(params[0]):
                scores[i, j] += (1/max_iters_for_avg) * clf.eval(n, d)

    plt.bar(np.linspace(1,len(params[0]),len(params[0]))-width, scores[0, :], width=width)
    plt.bar(np.linspace(1,len(params[0]),len(params[0])), scores[1, :], width=width)
    plt.bar(np.linspace(1,len(params[0]),len(params[0]))+width, scores[2, :], width=width)
    plt.xticks(ticks=np.linspace(1,len(params[0]),len(params[0])), labels=params[0])
    
    plt.show()
    