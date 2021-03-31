import numpy as np
from Sigmoid import sigmoid
from Sigmoid import sigmoid_gradient


# TODO Implement NN cost function for 2 layers neural network using regularization
# Return (J, grad); grad: "unrolled" vector of the partial derivatives of NN
def nn_cost_function(nn_params, input_layer_size,
                     hidden_layer_size, num_labels, X, y, lambda_r):

    # X: (m, input_layer_size); y: (m, 1) (Ex: (10, 10, 8, 4, 5,...))
    #     # Reshape nn_paras back into Theta1 and Theta2 (weight of layer 1 & 2)
    theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        newshape=(hidden_layer_size, input_layer_size + 1), order='F')
    theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                        newshape=(num_labels, hidden_layer_size + 1), order='F')

    # Set up initial variables & add bias unit
    # n1: input_layer_size, n2: hidden_layer_size, n3: output_layer_size
    m = X.shape[0]
    J = 0
    n3 = num_labels
    X = np.hstack((np.ones((m, 1)), X))  # X: (m, n1+1)

    # Initialize neural network parameters
    capital_delta1 = np.zeros(theta1.shape)
    capital_delta2 = np.zeros(theta2.shape)

    # Run along each example
    for i in range(m):
        # Forward propagation
        a1 = X[i]
        z2 = a1.dot(theta1.T)
        a2 = sigmoid(z2)
        a2 = np.hstack([1, a2])  # add bias unit
        z3 = a2.dot(theta2.T)
        a3 = sigmoid(z3)

        # Hypothesis function
        h = a3

        # Mapping y to binary matrix 0's and 1's (Ex: 2 -> [0, 1, 0])
        y_n3 = np.zeros((n3, 1))  # y is as K-dimensional vector
        y_n3[y[i, 0] - 1, 0] = 1

        # Compute cost function of each example
        j = (-np.dot(y_n3.T, np.log(h).T) - np.dot((1 - y_n3).T, np.log(1 - h).T))  # sum of K
        J = J + (j/m)  # Sum all cost of each example

        # Compute "Error" of each node in each layer
        delta3 = a3 - y_n3.T
        z2 = np.hstack([1, z2])
        delta2 = np.dot(theta2.T, delta3.T) * (sigmoid_gradient(z2).reshape(-1, 1))

        # Update delta of each layer l through each example
        capital_delta1 = capital_delta1 + (np.dot(delta2[1:, :], a1.reshape(1, -1)))
        capital_delta2 = capital_delta2 + (np.dot(delta3.T, a2.reshape(1, -1)))

    # Compute cost function with regularization
    sum1 = np.sum(np.sum(theta1[:, 1:] ** 2))
    sum2 = np.sum(np.sum(theta2[:, 1:] ** 2))
    J = J + (lambda_r / (2 * m)) * (sum1 + sum2)

    # Gradient (derivative) of cost with respect to Theta1
    theta1_grad = (1 / m) * (capital_delta1 + lambda_r * theta1)  # with regularization
    theta1_grad[:, 0] = ((1 / m) * capital_delta1)[:, 0]
    # Gradient (derivative) of cost with respect to Theta2
    theta2_grad = (1 / m) * (capital_delta2 + lambda_r * theta2)  # with regularization
    theta2_grad[:, 0] = ((1 / m) * capital_delta2)[:, 0]

    # Unroll the gradient
    grad = np.hstack((theta1_grad.ravel(order='F'), theta2_grad.ravel(order='F')))
    return J[0], grad


# TODO: Compute the gradient using "original equation" of derivative and
#  result a numerical estimate of the gradient (derivative of cost)
# Return gradient (derivative) of cost by using numerical equation (1) of derivative
def compute_numerical_gradient(theta, input_layer_size,
                               hidden_layer_size, num_labels, X, y, lambdas):
    e = 0.0001
    num_grad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)

    for p in range(len(theta)):
        perturb[p] = e
        # "_": shows "don't care" variables
        loss1, _ = nn_cost_function(theta - perturb, input_layer_size, hidden_layer_size,
                                    num_labels, X, y, lambdas)
        loss2, _ = nn_cost_function(theta + perturb, input_layer_size, hidden_layer_size,
                                    num_labels, X, y, lambdas)
        # Compute derivative by using original equation
        # (1) J_prime(x) = (J(x-h) - J(x+h)) / (2*e) with x: theta, h is equaled to e
        num_grad[p] = (loss2-loss1)/(2*e)
        perturb[p] = 0

    return num_grad

