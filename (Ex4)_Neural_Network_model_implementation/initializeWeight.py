import numpy as np


# TODO: Initialize weights with fixed (Can not change) dimension -> Useful for debugging (NOT DONE)
# "fan_in": # of nodes in layer "l"
# "fan_out": # of nodes in layer "l+1"
def debug_initialize_weight(fan_in, fan_out):
    W = np.zeros((fan_out, fan_in+1))
    W = np.reshape(range(len(W.ravel(order="F"))), W.shape)/10
    return W


# TODO: Initialize randomly weight of layer with L_in (#_nodes layer l) and L_out (#_nodes layer l+1))
def random_initialize_weight(L_in, L_out):
    # Confirm the size of weight parameters
    W = np.zeros((L_out, L_in+1))

    # Initialize random weight W in [-epsilon, epsilon]
    epsilon = 0.1
    W = np.random.rand(L_out, L_in+1) * (2*epsilon) - epsilon
    return W

# (NOTE) Difference between np.flatten() (1) & np.ravel() (2) (Both outputs the same values)
# "np.flatten()" always return a copy
# "np.ravel()": Return view of original array. If I modify array returned by ravel, it may modify the entries in the
# original array
