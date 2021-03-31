import numpy as np
from initializeWeight import debug_initialize_weight
from nnCostFunction import nn_cost_function
from nnCostFunction import compute_numerical_gradient


# TODO: Check correctness of computing gradient by comparing to numerical gradient
def check_NN_gradient(lambdas=0):
    # Including bias nodes
    input_layer_size = 3  # n1=3
    hidden_layer_size = 5  # n2=5
    num_labels = 3  # n3=3
    m = 5

    # Initialize "random" test theta
    # (NOTE) debug_initialize_weight(): already adds bias to Theta
    Theta1 = debug_initialize_weight(hidden_layer_size, input_layer_size)  # Theta1: (n2, n1+1)
    Theta2 = debug_initialize_weight(num_labels, hidden_layer_size)  # Theta2: (n3, n2+1)

    # Initialize X and y
    X = debug_initialize_weight(input_layer_size-1, m)  # X: (m, n1) = (5,3)
    # (NOTE) .reshape(-1, 1): a way to confirm 2D array (without knowing # of rows)
    # (NOTE) .mod(arr1, arr2) return arr1 % arr2 (modulo)
    y = 1 + np.mod(range(m), num_labels).reshape(-1, 1)  # y: (m, 1)

    # Unroll parameters
    # "order="F"": flatten in column major order; "order="C"": flatten in row major order
    nn_params = np.hstack((Theta1.ravel(order="F"), Theta2.ravel(order="F")))

    # Create shortcut for the cost function
    (J_regu, grad) = nn_cost_function(nn_params, input_layer_size,
                               hidden_layer_size, num_labels, X, y, lambdas)

    # Compute gradient using numerical equation
    numgrad = compute_numerical_gradient(nn_params, input_layer_size, hidden_layer_size,
                                         num_labels, X, y, lambdas)
    print("The 2 columns must be very similar..")
    for i, j in zip(grad[:10], numgrad[:10]):
        print(i, j)

    # Compute norm of diff between "grad" and "numgrad"
    diff = np.linalg.norm(numgrad-grad) / np.linalg.norm(numgrad+grad)  # (2)
    if diff < 0.00000001:
        print("\nBackpropagation is correct")
    else:
        print("\nBackpropagation is not correct")



    # (NOTE) np.linalg.norm() compute the norm of an array or vector
    # (NOTE) a=[[2,3], [4,3]] -> np.linalg.norm(a) = sqrt(2^2 + 3^2 + 4^2 + 3^2)
    # (NOTE) (2) is a way of checking the SIMILARITY of 2 columns
