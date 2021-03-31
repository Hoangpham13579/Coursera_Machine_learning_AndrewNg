import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from linearRegCost import linearRegCostFunction
from trainLinearReg import trainLinearReg
from learningCurve import learningCurve
from polyMappingFeature import poly_features, feature_normalize
from polyLearningCurve import poly_learning_curve
from validationCurve import validation_curve


############### (1) Loading and visualizing data ###################
print("Loading train/validation/test data set")

# Load training data
data = scipy.io.loadmat("ex5data1.mat")
X = data["X"]  # X: (12, 1)
y = data["y"]  # y: (12, 1)
X_vali = data["Xval"]  # X_vali: (21, 1)
y_vali = data["yval"]  # X_vali: (21, 1)
X_test = data["Xtest"]  # X_test: (21, 1)
y_test = data["ytest"]  # y_test: (21, 1)
m = X.shape[0]
X = np.hstack((np.ones((m, 1)), X))  # X: (m, n+1)

# Visualize training data set
print("\nVisualize training data set")
plt.plot(X[:, 1:], y, "x", color="red", markersize=15)
plt.xlabel("Change in water level (x)")
plt.ylabel("Water flowing out of dam (y)")
plt.show()

input("Press Enter to continue...")  # Stop for a moment


################ (2) Regularized linear regression cost #################
print("\nCompute cost function for regularized linear regression")

# Initialize parameter & add bias to X
theta = np.ones(shape=(X.shape[1], 1))

# Compute cost function
J, _ = linearRegCostFunction(theta, X, y, 1)

print(f'Cost at theta = [1 ; 1]: {J} \n(this value should be about 303.993192)')
input("Press Enter to continue...")  # Stop for a moment


################## (3) Regularized linear regression Gradient #################
print("\nCompute Gradient function for regularized linear regression")

# Initialize parameter & add bias to X
theta = np.ones(shape=(X.shape[1], 1))

# Compute cost function
(J, grad) = linearRegCostFunction(theta, X, y, 1)

print(f'Gradient at theta = [1 ; 1]: {grad[0, 0]}, {grad[1, 0]} '
      '\n(this value should be about [-15.303016; 598.250744])\n')
input("Press Enter to continue...")  # Stop for a moment


################# (4) Train linear regression ################
print("\nVisualize hypothesis lines trained by training data set.")
# Train linear regression with lambdas = 0
lambdas = 0
train_theta = trainLinearReg(X, y, lambdas)

# Plot hypothesis line fits to the data
plt.plot(X[:, 1:], y, "x", color="red", markersize=10)
plt.xlabel("Change in water level (x)")
plt.ylabel("Water flowing out of dam (y)")
plt.plot(X[:, 1], np.dot(X, train_theta), "--", color="blue", linewidth=3)
plt.show()
input("Press Enter to continue...")  # Stop for a moment


################### (5) Learning curve for linear regression #############
# Learning curve: represent Errors values of training and cross validation
# depending on each number of training example (from 1 to m)
print(f"\nVisualize learning curve for linear regression")
# Initialize parameter
lambdas = 0
X_vali = np.hstack((np.ones((X_vali.shape[0], 1)), X_vali))  # X_vali: (m_vali, n+1)
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))  # X_test: (m_test, n+1)

# Compute train and cv errors set
(error_train, error_val) = learningCurve(X, y, X_vali, y_vali, lambdas)

# Visualize train and cross validation ERRORS set
plt.plot(range(1, m+1), error_train, "-", color="blue")
plt.plot(range(1, m+1), error_val, "-", color="red")
plt.xlabel("Number of training example")
plt.ylabel("Cost function value")
plt.legend(("Training error", "Cross validation error"))
plt.show()

input("Press Enter to continue...")  # Stop for a moment


#################### (6) Feature mapping for Polynomial Regression ####################
print(f"\nMapping from linear regression to polynomial regression")
p = 8

# Mapping X from linear to p's polynomial & normalize
X_poly = poly_features(data["X"].reshape(X.shape[0]), p)
X_poly, mu, sigma = feature_normalize(X_poly)  # Normalize X -> X_poly: (m, p)
X_poly = np.hstack((np.ones((X_poly.shape[0], 1)), X_poly))  # X_poly: (m, p+1)

# Mapping X_test from linear to p's polynomial & normalize
X_poly_test = poly_features(data["Xtest"].reshape(X_test.shape[0]), p)
X_poly_test, _, _ = feature_normalize(X_poly_test)  # Normalize X
X_poly_test = np.hstack((np.ones((X_poly_test.shape[0], 1)), X_poly_test))  # X_poly_test: (m_test, p+1)

# Mapping X_vali from linear to p's polynomial & normalize
X_poly_vali = poly_features(data["Xval"].reshape(X_vali.shape[0]), p)
X_poly_vali, _, _ = feature_normalize(X_poly_vali)  # Normalize X
X_poly_vali = np.hstack((np.ones((X_poly_vali.shape[0], 1)), X_poly_vali))  # X_poly_vali (m_vali, p+1)

print(f"Normalized training example 1:\n{X_poly[1, :]}")
input(f"Press Enter to continue...")  # Stop for a moment


############## (7) Learning curve for the polynomial regression ###############
print(f"\nPlot learning curve for polynomial regression without regularization")
# Without regularization
lambdas = 0
poly_learning_curve(X, X_poly, y, X_poly_vali, y_vali, lambdas, p)

input(f"Press Enter to continue...")  # Stop for a moment

# With regularization
print(f"\nPlot learning curve for polynomial regression with regularization (lambdas=1)")
lambdas = 1
poly_learning_curve(X, X_poly, y, X_poly_vali, y_vali, lambdas, p)

# As a result, regularization help to solve overfitting problem by look at second plot
# Error of train and cv tend to be right on when m=4
input(f"Press Enter to continue...")  # Stop for a moment


################ (8) Select lambdas on Validation dataset #################
# Select best lambda based on polynomial regression by computing error of train and validation dataset
print(f"\nTest various value of lambdas on validation dataset --> Select the best lambda")
# Compute error of train and validation dataset based on diff value of lambda
lambda_vec, error_train, error_vali = validation_curve(X_poly, y, X_poly_vali, y_vali)

# Visualize error corresponding to value of lambdas
plt.plot(lambda_vec, error_train, color="blue", linewidth=2)
plt.plot(lambda_vec, error_vali, color="red", linewidth=2)
plt.xlabel("Lambda")
plt.ylabel("Error (cost function) values")
plt.legend(("Training set error", "Validation set error"))
plt.show()

# Showing error (Cost value) result relating to lambdas value
print(f"Lambda\tTraining error\tValidation error")
for i in range(len(lambda_vec)):
      print(f'{lambda_vec[i]}\t{error_train[i]}\t{error_vali[i]}')

# Result: as lambdas = 1, the error of training and validation dataset are nearly the same (Solving overfitting)
# --> lambda = 1 is the best value to apply for regularization
# (NOTE) Error: cost value (Not diff between y and predicting val)
