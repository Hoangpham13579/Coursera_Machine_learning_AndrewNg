import numpy as np
from sklearn import svm
from gaussianKernel import gaussian_kernel_matrix


# TODO: Compute optimal C and sigma learning parameters using cross validation set
# Return: optimal C and sigma param based on cross-validation set
def svm_kernel_param(X, y, Xval, yval):
    # Initialize parameters
    C = 1
    sigma = 3

    # Showing dataset
    allC = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    allSigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    bestC = allC[0]
    bestSigma = allSigma[0]
    previousErr = 1000

    # Finding the best C and Sigma value
    for c in allC:
        # currentC = c
        for sigma in allSigma:
            # currentSigma = sigma

            # Training the SVM model on the X training data
            # Why "precomputed"? -> When using "f" value in training dataset (Gaussian kernel)
            clf = svm.SVC(kernel="precomputed", C=c)  # Construct SVM model
            # Set "l" value (landmarks) equal to "X" as instruction -> Compute "f" value in SVM
            f_val = gaussian_kernel_matrix(X, X, sigma=sigma)
            clf.fit(f_val, y)  # Fit data to model for training

            # Predict Y based Xval & Compute mean of ERROR
            f_val_pred = gaussian_kernel_matrix(Xval, X)
            y_pred = clf.predict(f_val_pred)
            err_mean = np.mean(y_pred != yval)

            if err_mean < previousErr:
                bestC = c
                bestSigma = sigma
                previousErr = err_mean

    return bestC, bestSigma




