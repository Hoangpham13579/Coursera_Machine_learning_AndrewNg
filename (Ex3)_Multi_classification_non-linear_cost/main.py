import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from displayData import display_data
from oneVsAll import oneVsAll
from predict import predictOneVsAll

# (TYPE) Using logistic regression (existing only 1 theta array) to train model
# with large amount of features --> Disadvantage: It's really slow as number of features increase


# Set up some need parameters
input_layer_size = 400  # 20*20 input image of digits
num_labels = 10  # 10 labels, from 1 to 10
                 # (NOTE) 10 is mapped by 0 (Only in visualization NOT in y)

########### (1) Load and visualizing data ###############
# (ex3data1.mat): each example is an array (400 element -> 20*20) representing an image of a label
# Load training data
data = scipy.io.loadmat("ex3data1.mat")
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


############ (2) One vs all training ###############
# (PURPOSE) Minimize cost to result an optimal all_theta 2D array
print("Train One-vs-all logistic regression")

# Initial parameters
lambdas = 0.1

# Training theta
all_theta = oneVsAll(X, y, num_labels, lambdas)  # (num_labels, 401)
all_theta = all_theta.reshape((num_labels, n+1))

input("Press Enter to continue...")  # Stop for a moment


############## (3) Predict One-Vs_All model #################
# Compute the prediction for y based on trained theta
p = predictOneVsAll(all_theta, X)

# Compute accuracy
print("Train accuracy: ", np.mean(p == y) * 100)



