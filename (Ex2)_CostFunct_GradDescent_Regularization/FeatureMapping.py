import numpy as np


# TODO generating features for "Regularization" technique solving "Overfitting"
def features_mapping(X1, X2):
    # Input: 2 independent features
    # Output: X1, X2, X1^2, X2^2, X1*X2, X1*X2^2, etc...

    degree = 6
    # Alternative: (degree+1)*(degree+2)/2
    mapping = np.ones(( X1.shape[0], sum(range(degree+2)) ))
    count = 1

    for i in range(1, degree+1):
        for j in range(i+1):
            mapping[:, count] = np.power(X1, (i-j)) * np.power(X2, j)
            count += 1

    return mapping
