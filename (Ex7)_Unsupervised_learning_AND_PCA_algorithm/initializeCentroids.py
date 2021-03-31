import numpy as np


def initialize_centroids(X, K):
    # X: an 2D array; K: int
    # Choose randomly K example from X
    initial_centroids = X[np.random.choice(X.shape[0], K), :]
    return initial_centroids

