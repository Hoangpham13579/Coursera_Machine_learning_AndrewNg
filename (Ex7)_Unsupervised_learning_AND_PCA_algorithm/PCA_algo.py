import numpy as np


# TODO: Perform the PCA algorithm
# Output of svd function
def pca(X):
    # Compute covariance of dataset
    m = X.shape[0]
    cov_matrix = (1/m) * X.T.dot(X)

    # Calculate the result svd function
    U, S, _ = np.linalg.svd(cov_matrix)
    return U, S


