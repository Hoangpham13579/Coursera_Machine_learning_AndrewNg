import numpy as np
from Sigmoid import sigmoid


# TODO: Making the prediction based on X and trained NN parameters
def prediction(Theta1, Theta2, X):
    # Initialize parameters
    m, n = X.shape

    # Making prediction
    z2 = np.dot(np.hstack((np.ones((m, 1)), X)), Theta1.T)
    h2 = sigmoid(z2)  # h2: (m, n2)
    z3 = np.dot(np.hstack((np.ones((m, 1)), h2)), Theta2.T)
    h3 = sigmoid(z3)  # h3: (m, n3)
    p = np.argmax(h3, axis=1)+1  # p: (m, 1)

    return p

