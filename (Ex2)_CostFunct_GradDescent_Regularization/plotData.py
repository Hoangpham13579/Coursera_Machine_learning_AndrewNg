import matplotlib.pyplot as plt
import numpy as np


# TODO: Plot classification dataset with 2 features
def plotdata(X, y, ax=None, **plt_kwargs):
    # Initial values
    if ax is None:
        ax = plt.gca()

    # Differentiate "admit" and "not admit" examples
    pos = np.where(y == 1)
    neg = np.where(y == 0)

    # Plotting dataset
    ax.scatter(X[pos, 0], X[pos, 1], marker="x", color="black")
    ax.scatter(X[neg, 0], X[neg, 1], marker="x", color="red")
    ax.set_xlabel("1st col data")
    ax.set_ylabel("2nd col data")
    ax.legend(["admitted", "not admitted"])

    return ax

