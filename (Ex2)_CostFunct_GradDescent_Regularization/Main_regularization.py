import numpy as np
import matplotlib.pyplot as plt
from plotData import plotdata
from FeatureMapping import features_mapping
from CostFunction import cost_logistic_regularization
from PredictFunction import logistic_prediction
import scipy.optimize as opt


# Load dataset
data2 = np.loadtxt("data/ex2data2.txt", delimiter=",")
X = data2[:, :2]
y = data2[:, -1]

# (1) Plot dataset
plt.figure(figsize=(5, 5))
plotdata(X, y)
plt.xlabel("1st col data")
plt.ylabel("2nd col data")
plt.legend(["admitted", "not admitted"])
plt.show()

input("Press Enter to continue...")  # Stop for a moment


################### (2) Regularize logistic regression #############
# Mapping polynomial features (Ex: (x1,x2) -> x1, x2, x1^2, x2^2, x1*x2,...)
# (NOTE) FeatureMapping also add bias col for X
X = features_mapping(X[:, 0], X[:, 1])

# Initialize theta & lambda
# init_theta = np.zeros((X.shape[1], 1))
init_theta = np.zeros(X.shape[1])
lambdas = 1
# X: (m, 29); init_theta: (29, 1), y: (m, 1) -> (n+1) = 29

# Guarantee the shape of variables
m = X.shape[0]
y = np.reshape(y, (m, 1))

# Compute initial cost and grad for regularized logistic regression
[J, grad] = cost_logistic_regularization(init_theta, X, y, lambdas)
print("Cost Function with init theta: ", J)
print("Expected cost (approx): 0.693")
print("Grad using init theta (1st 5 values only): \n", grad[:5])
print("Expected value: \t0.0085\t 0.0188\t 0.0001\t 0.0503\t 0.0115\n")

# Compute and display cost and gradient
# with all-vs-ones theta and lambda = 10
# test_theta = np.ones((X.shape[1], 1))
test_theta = np.ones(X.shape[1])
[J_test, grad_test] = cost_logistic_regularization(test_theta, X, y, 10)

print('\nCost at test theta (with lambda = 10): ', J_test)
print('Expected cost (approx): 3.16')
print('Gradient at test theta - first five values only: ', grad_test[:5])
print('Expected gradients (approx) - first five values only:')
print(' 0.3460\t 0.1614\t 0.1948\t 0.2269\t 0.0922')

input('Program paused. Press enter to continue')


############# (3) Optimize theta with regularization & compute accuracy ########
# Task: Try different value of lambda (0, 1, 10, 100) & see how it affect the decision
# Initialize fitting parameters
m = X.shape  # X: (m, 29) _ n+1=29
# (NOTE) Initialize weight as 1D & Change it to 2D in cost function
init_weight = np.zeros(X.shape[1])  # 1D array

# Set regularization lambda to 1
lambdas = 1

# Optimize cost function and get theta
result = opt.fmin_tnc(func=cost_logistic_regularization,
                    # (NOTE) x0 must be 1D array (n+1, )
                    x0=init_weight, args=(X, y, lambdas))

trained_theta = result[0]
# (NOTE) It's a must to confirm the DIM of trained_theta
trained_theta = np.reshape(trained_theta, (trained_theta.shape[0], 1))

# Compute accuracy in our training
p = logistic_prediction(trained_theta, X)

print("Train accuracy: ", np.mean(p == y) * 100)
print("Expected accuracy (with lambdas = 1): 83.1 (approx)")

