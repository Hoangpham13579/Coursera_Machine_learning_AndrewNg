import numpy as np


# TODO: Reduce n-dim data X to K-dim data Z
def project_data(X, U, K):
    U_reduce = U[:, :K]  # (n, K)
    Z = np.dot(X, U_reduce)  # (m, K)
    return Z

