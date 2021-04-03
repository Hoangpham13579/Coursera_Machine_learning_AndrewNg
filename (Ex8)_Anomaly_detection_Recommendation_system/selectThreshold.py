import numpy as np


# TODO: Choose the most appropriate epsilon (threshold of "p") for the value of Gaussian (normal) distribution (p)
# Return: Best value of epsilon & Best value of F1 score
# Input: yval, pval: y and p value of validation dataset
def select_threshold(yval, pval):
    # Initialize parameter
    best_f1 = 0
    best_epsilon = 0
    # "step" the distance of each checking epsilon
    step = (np.max(pval) - np.min(pval)) / 1000

    for epsilon in np.arange(np.min(pval), np.max(pval), step):
        # y_pred detects that data example anomaly (outlier) (= 0) or not (= 1)
        y_pred = pval < epsilon
        # tp: true positive; fp: false positive; fn: false negative
        tp = np.sum(np.logical_and(yval == 1, y_pred == 1))
        fp = np.sum(np.logical_and(yval == 0, y_pred == 1))
        fn = np.sum(np.logical_and(yval == 1, y_pred == 0))

        # Compute "precision" and "recall" and "F1 score"
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2*precision*recall) / (precision + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon
    return best_epsilon, best_f1


# "np.logical_and(x1, x2)": comparing the logical values of x1 and x2 element-wise

