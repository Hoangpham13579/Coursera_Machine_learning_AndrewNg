import numpy as np


# TODO: Reconstructing compressed data approximately to original data
def recover_data(Z, U, K):
    # Z: (m, K)
    U_reduce = U[:, :K]  # (n, K)
    X_approx = np.dot(Z, U_reduce.T)
    return X_approx

