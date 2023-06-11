# plots
from matplotlib import pyplot as plt
import numpy as np

plt.style.use('seaborn')

topic_mapping = {
    'Bussiness': 0,
    'Education': 1,
    'Entertainment': 2,
    'News': 3,
    'Football': 4
    }

def visualize_dataset(features=[], labels=[]):
    timestamp, comments, likes, shares, values = features[:, 0], features[:, 1],features[:, 2],features[:, 3],features[:, 4]
    topics = labels
    values = np.array([0 if values[i] < 0 else 1 for i in range(len(values))])
    ticks = list(np.unique(topics))

    # fig, [ax1, ax2, ax3] = plt.subplots(nrows=3, ncols=1, sharex=True)
    # mpl1 = ax1.scatter(likes, shares, s=5+20*values, c=topics, cmap='hsv')
    # mpl2 = ax2.scatter(likes, comments, s=5+20*values, c=topics, cmap='hsv')
    # mpl3 = ax3.scatter(comments, comments, s=5+20*values, c=topics, cmap='hsv')
    # cbar1 = fig.colorbar(mpl1)
    # cbar2 = fig.colorbar(mpl2)
    # cbar3 = fig.colorbar(mpl3)
    # cbar1.set_ticks(ticks=ticks, labels=topic_mapping)
    # cbar2.set_ticks(ticks=ticks, labels=topic_mapping)
    # cbar3.set_ticks(ticks=ticks, labels=topic_mapping)
    # ax1.set_title('facebook posts dataset visualisation')
    # ax3.set_xlabel('timestamp (Ascending)')
    # ax1.set_ylabel('likes')
    # ax2.set_ylabel('shares')
    # ax3.set_ylabel('comments')
    # fig.tight_layout()

    fig = plt.figure(1, figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(likes, comments, shares, c=topics, cmap=plt.cm.Set1, edgecolor="k", s=5+20*values)

    ax.set_title("facebook posts dataset 3D visualisation")
    ax.set_xlabel("likes")
    ax.xaxis.set_ticklabels([])
    ax.set_ylabel("comments")
    ax.yaxis.set_ticklabels([])
    ax.set_zlabel("shares")
    ax.zaxis.set_ticklabels([])

    plt.show()