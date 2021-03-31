import numpy as np


# TODO: Compute the mean of all data points closed to each centroid (1)
# Return : set of new centroids (Computed by (1))
def compute_centroid_mean(X, idx, K):
    # Initial parameter (X: (m,n), idx: (m, 1))
    centroids = np.zeros((K, X.shape[1]))

    # Compute the centroid mean
    for k in range(K):
        centroids[k, :] = np.mean(X[(idx == k+1).T[0], :], axis=0)
    return centroids


