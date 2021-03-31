import numpy as np
from Sigmoid import sigmoid


# TODO: Prediction using forward propagation (including output layer)
# Return p: vec of predicted labels of each example in X given the trained weight
def predict(theta1, theta2, X):
    # n1: #_nodes layer 1; n2: #_nodes layer 2; n3: #_nodes layer 3
    # X: (m, n1); theta1: (n2, n1+1); theta2: (n3, n2+1)
    m = X.shape[0]  # Number of examples

    # Guarantee the size of parameter
    p = np.zeros((m, 1))

    # Forward propagation
    a1 = np.hstack((np.ones((m, 1)), X))  # a1: (m, n1+1)
    z1 = np.dot(a1, theta1.T)  # z1: (m, n2)
    a2 = np.hstack((np.ones((m, 1)), sigmoid(z1)))  # a2: (m, n2+1)
    z2 = np.dot(a2, theta2.T)  # z2: (m, n3)
    a3 = sigmoid(z2)  # a3: (m, n3)

    # Choosing optimal label depending on trained theta & output layer
    p = np.argmax(a3, axis=1)+1
    p = np.reshape(p, (m, 1))
    return p  # return p: (m, 1)


# TODO: Predict with n layers (all_theta including all thetas of each layers)
# Return: p an 2D array in which each row predicted vec for each examples
# (Ex: [1,0,0]: car; [0,1,0]: motor; [0,0,1]: bicycle) -> num_label = 3
# (NOTE) all_theta: a matrix where i-th row is a trained logistic theta vector for the i-th labels (predicted labels)
def predictOneVsAll(all_theta, X):
    # all_theta: (num_label, n+1)
    # Initialize parameter
    m = X.shape[0]  # Number of examples
    num_labels = all_theta.shape[0]  # Total number of labels

    # Guarantee the size of parameter & Add bias col to X
    p = np.zeros((m, 1))
    X = np.hstack((np.ones((m, 1)), X))  # X: (m, n+1)

    # Compute predicting 2D array of each example
    prediction = sigmoid(np.dot(X, all_theta.T))  # prediction: (m, num_label)

    # Compute the predicted label for each examples
    p = np.argmax(prediction, axis=1)+1
    p = p.reshape((m, 1))
    return p  # return p: (m, 1)


# (NOTE) np.argmax() return indices of max value along an array



