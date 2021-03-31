import numpy as np
import matplotlib.pyplot as plt
from displayData import display_data
import scipy.io
from nnCostFunction import nn_cost_function
from Sigmoid import sigmoid_gradient
from initializeWeight import random_initialize_weight
from checkNNGradient import check_NN_gradient
import scipy.optimize as opt
from predict import prediction


# Set up needed parameter
input_layer_size = 400  # 20*20 inputs images of digits
hidden_layer_size = 25  # 25 hidden units
num_labels = 10  # 10 labels from 1 to 10
                 # Note that 10 is mapped by 0 in visualization

############### (1) Loading and visualizing data ###################
print("Loading and visualizing data")

# Load training data
data = scipy.io.loadmat("ex4data1.mat")
X = data["X"]  # (5000, 400)
y = data["y"]  # (5000, 1)
(m, n) = X.shape

# Visualize random 100 sample digits from data set (display_data())
fig = plt.figure(figsize=(5, 5))
ax = display_data(X, y, fig)
ax.tick_params(
    axis="both", which="both",
    bottom=False, left=False,  # Remove ticks alon bottom, left edge
    labelbottom=False, labelleft=False  # Remove labels along bottom and left edge
)
plt.show()
input("Press Enter to continue...")  # Stop for a moment


######### (2) Loading parameters ################
# Install some pre-initialized neural network parameters
print("\nLoading saving theta parameters...")
initialTheta = scipy.io.loadmat("ex4weights.mat")
Theta1 = initialTheta["Theta1"]
Theta2 = initialTheta["Theta2"]

# Unrolled parameters
nn_params = np.hstack((Theta1.ravel(order="F"), Theta2.ravel(order="F")))
input("Press Enter to continue...")  # Stop for a moment


################ (3) Compute cost (with and without regularization) ################
print("\nChecking cost function (with regularization)")

# Weight of regularization parameter
lambdas = 1

J, _ = nn_cost_function(nn_params, input_layer_size,
                     hidden_layer_size, num_labels, X, y, lambdas)

print("Cost at parameter (ex4weights.mat): ", J)
print("This value should be 0.383770")
input("Press Enter to continue...")  # Stop for a moment


################ (4) Sigmoid gradient #################
print("\nEvaluating the gradient of sigmoid...")

g = sigmoid_gradient(np.array([-1, -0.5, 0, 0.5, 1]))
print("Sigmoid gradient evaluated from [-1, -0.5, 0, 0.5, 1] is: \n", g)
input("Press Enter to continue...")  # Stop for a moment


################### (5) Initialize parameters #################
# Using function debugInitializeWeights to initialize theta
print("\nInitializing neural network parameters...")

initial_theta1 = random_initialize_weight(input_layer_size, hidden_layer_size)
initial_theta2 = random_initialize_weight(hidden_layer_size, num_labels)

# Unrolled parameters
initial_nn_params = np.hstack((initial_theta1.flatten(), initial_theta2.flatten()))

input("Press Enter to continue...")  # Stop for a moment


##################### (6) Implement backpropagation ################
print("\nChecking backpropagation...")

# Check correctness of backpropagation by running checkNNGradient
check_NN_gradient()

input("Press Enter to continue...")  # Stop for a moment


####################### (7) Implement Regularization ###################
print("\nChecking backpropagation with regularization...")

# Check correctness of backpropagation by running checkNNGradient
lambdas = 3
check_NN_gradient(lambdas)

# Check cost function with regularization
print("\nCheck the correctness of cost function with regularization")
J_check, _ = nn_cost_function(nn_params, input_layer_size, hidden_layer_size,
                              num_labels, data["X"], data["y"], lambdas)
print("Cost values at lambdas = 3 with regularization is: ", J_check)
print("Expected value: 0.576051")
input("Press Enter to continue...")  # Stop for a moment


####################### (8) Train NN model ####################
print("\nTrain Neural Network model...")
lambdas = 1

# Train NN models
result = opt.minimize(nn_cost_function, initial_nn_params,
                      args=(input_layer_size, hidden_layer_size,
                            num_labels, data["X"], data["y"], lambdas),
                      method="L-BFGS-B", jac=True, options={"maxiter": 50})

# Obtain Theta1, Theta2 back from trained_params
train_nn_params = result["x"]
train_theta1 = np.reshape(train_nn_params[:hidden_layer_size*(input_layer_size+1)],
                    (hidden_layer_size, input_layer_size+1), order="F")
train_theta2 = np.reshape(train_nn_params[hidden_layer_size*(input_layer_size+1):],
                    (num_labels, hidden_layer_size+1), order="F")

input("Press Enter to continue...")  # Stop for a moment


################### (9) Making prediction ######################
print("\nCompute Accuracy of model...")
# Making the prediction
p_pred = prediction(train_theta1, train_theta2, data["X"])

# Compute accuracy depending on trained_weights
print("Model accuracy: ", np.mean(p_pred == y.T) * 100)

