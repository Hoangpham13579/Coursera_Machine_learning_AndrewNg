import numpy as np
import matplotlib.pyplot as plt
from plotData import plotdata
from CostFunction import cost_function_logistic
import scipy.optimize as opt  # Find optimal weights by minimizing cost
from PredictFunction import logistic_prediction


# Load dataset
data = np.loadtxt("data/ex2data1.txt", delimiter=",")
X = data[:, :2]
y = data[:, -1]

# (1) Plot dataset
plt.figure(figsize=(5, 5))
plotdata(X, y)
plt.xlabel("1st col data")
plt.ylabel("2nd col data")
plt.legend(["admitted", "not admitted"])
plt.show()

input("Press Enter to continue...")  # Stop for a moment


##################### (2) Compute cost and gradient descent #################
# Initial value
(m, n) = X.shape  # n=2; m=100
X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)  # Add bias to X
initial_theta = np.zeros(n+1)  # 1D weight (n+1, )

# (NOTE_Must) Guarantee the shape of parameters
y = np.reshape(y, (m, 1))
X = np.reshape(X, (m, n+1))

# Compute cost and gradient
(J, grad) = cost_function_logistic(initial_theta, X, y)
# grad = gradient_logistic(X, y, initial_theta)

print("Cost function: ", J)
print("Expected cost (approx): 0.693\n")
print("Gradient at initial theta (zeros): \n", grad)
print("Expected gradients (approx):\n -0.1000\t -12.0092\t -11.2628\t")

input("Press Enter to continue...")  # Stop for a moment


############### (3) Optimizing theta (1 labels model) #######################
# initial_theta: (n+1, 1), X: (m, n+1); y: (m, 1)
# Minimize cost function (J) and find the optimal theta value
result = opt.fmin_tnc(func=cost_function_logistic,
                    # (NOTE) x0 must be 1D array (n+1, )
                    x0=initial_theta, args=(X, y))
trained_theta = result[0]
# (MUST) Confirm the DIM of trained theta 2D
trained_theta = np.reshape(trained_theta, (trained_theta.shape[0], 1))

# Cost function using trained theta
(J_trained, grad_trained) = cost_function_logistic(trained_theta, X, y)

print("Cost at trained theta : ", J_trained)
print("Expected cost (approx): 0.203\n")
print("Trained theta: \n", trained_theta)
print("Expected theta (approx):")
print(" -25.161\t 0.206\t 0.201\t")

input("Press Enter to continue...")  # Stop for a moment


################ (4) Predict and Compute accuracies #################
# Tasks: Predict label (0 or 1) using trained theta & Computes the Accuracies
p = logistic_prediction(trained_theta, X)

print("Train accuracy: ", np.mean(p == y) * 100)
print("Expected accuracy (approx): 89.0")



# (NOTE) 1st para of cost function must be theta --> to activate
# optimized function opt.fmin_tnc()
