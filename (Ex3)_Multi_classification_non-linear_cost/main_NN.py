import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from displayData import display_data
from predict import predict

# Set up some need parameters
input_layer_size = 400  # 20*20 input image of digits
num_labels = 10  # 10 labels, from 1 to 10
                 # (NOTE) 10 is mapped by 0 (Only in visualization NOT in y)


########### (1) Load and visualizing data ###############
# (ex3data1.mat): each example is an array (400 element -> 20*20) representing an image of a label
print("Loading and visualizing data")

# Load training data
data = scipy.io.loadmat("data/ex3data1.mat")
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


################ (2) Loading parameters ####################
# Install some pre-initialized neural network parameters
print("Loading saving theta parameters...")
initialTheta = scipy.io.loadmat("data/ex3weights.mat")
theta1 = initialTheta["Theta1"]
theta2 = initialTheta["Theta2"]


################## (3) Implement the prediction #################
# Predict using forward propagation
pred = predict(theta1, theta2, X)

# Compute accuracy
print("Train accuracy: ", np.mean(pred == y) * 100)  # 97.52%

