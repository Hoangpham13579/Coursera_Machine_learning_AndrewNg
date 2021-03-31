import numpy as np


# TODO: Compute similarity of X (LINEAR vec) when applying kernels
# Input: x1: linear training dataset; x2: linear landmarks (value "l")
# Return: "f" value (similarity) when apply Kernel tricks
def gaussian_kernels(x1, x2, sigma):
    # Initialize parameter
    x1 = x1.ravel()
    x2 = x2.ravel()

    # Compute similarity between x1 and x2 using a Gaussian Kernels
    sim = np.exp(-np.sum((x1-x2)**2) / (2*sigma**2))
    return sim


# TODO: Set value of "l" (landmarks) the same as value of X (Non-linear training dataset -> Matrix)
# Return: Gaussian kernels matrix "f" when X1 & X2 (X AND l) are both MATRIX (Important)
def gaussian_kernel_matrix(X1, X2, sigma=0.1):
    # Initialize parameters
    gaussian_matrix = np.zeros((X1.shape[0], X2.shape[0]))

    for i, x1 in enumerate(X1):  # "x1" is each row with index "i" in matrix X1
        for j, x2 in enumerate(X2):
            gaussian_matrix[i, j] = gaussian_kernels(x1, x2, sigma)

    return gaussian_matrix

