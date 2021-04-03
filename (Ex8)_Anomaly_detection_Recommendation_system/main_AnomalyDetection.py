import numpy as np
import scipy.io
import seaborn as sns
import matplotlib.pyplot as plt

from gaussianDistribution import estimate_gaussian_para
from gaussianDistribution import multivariate_gaussian
from fitMGDToVisualization import fit_MGD_visualization
from selectThreshold import select_threshold


####################### (1) Anomaly detection ##########################
# Loading data
data1 = scipy.io.loadmat("data/ex8data1.mat")
# print(data1.keys())  # data1 contains X, Xval, yval (Xval: X validate dataset)

# Visualize data set data1
print("Visualizing X data")
ax = sns.scatterplot(data1["X"][:, 0], data1["X"][:, 1], marker="x", color="b", s=25)
ax.set(xlabel="Latency (ms)", ylabel="Throughput (mb/s)", title="1st data set")
plt.show()
input("Program pause, Press enter to continue")


# Compute Gaussian para & Multivariate Gaussian value
print("\nCompute Multivariate Gaussian (Normal) Distribution (MGD) for each data example")
mu, var = estimate_gaussian_para(data1["X"])
p = multivariate_gaussian(data1["X"], mu, var)
print(f"Shape of X and p (MGD result) simultaneously:\n {data1['X'].shape}\t{p.shape}")

# Apply result of MGD to visualization of data
print("\nVisualize the fit Multivariate Gaussian (Normal) Distribution to the dataset X")
fit_MGD_visualization(data1["X"], mu, var)
plt.show()
input("Program pause, Press enter to continue")


# Select the threshold for the validation dataset
# Compute the Gaussian (normal) distribution value for validation dataset
print("\nCompute the most suited threshold (epsilon) and their appropriate f1 score")
pval = multivariate_gaussian(data1["Xval"], mu, var)

# Compute the most suited threshold (epsilon) & their related F1 score
epsilon, f1 = select_threshold(data1["yval"].ravel(), pval)

print(f"Best epsilon found using cross-validation: {epsilon}")
print(f"Best F1 on Cross Validation Set: {f1}")
print("(you should see a value epsilon of about 8.99e-05)")
print("(you should see a Best F1 value of  0.875000)")
input("Program pause, Press enter to continue")


# Finding the outliers in dataset based on epsilon
print("\nVisualize the outliers inside the dataset")
outliers = np.argwhere(p < epsilon)
outliers = outliers.T[0]

# Visualize the outliers in dataset
ax = fit_MGD_visualization(data1["X"], mu, var)
# "facecolors": only plot the border of data value circle
ax.scatter(data1["X"][outliers, 0], data1["X"][outliers, 1], facecolors="none", color="r", s=200)
ax.set(title="Classify anomaly data sample")
plt.show()
input("Program pause, Press enter to continue")


########################## () High dimensional dataset detection #################
# Input "ex8data2.mat"
print("\nDetect anomaly data point in high dimensional dataset")
data2 = scipy.io.loadmat("data/ex8data2.mat")
# data2 contains X, Xval, yval (validation dataset) (with n number of features)

# Compute Gaussian (normal distribution) value for each data example in both training and validate dataset
mu, var = estimate_gaussian_para(data2["X"])
p = multivariate_gaussian(data2["X"], mu, var)
pval = multivariate_gaussian(data2["Xval"], mu, var)

# Finding the best value of epsilon
epsilon, f1 = select_threshold(data2["yval"].ravel(), pval)

print(f"Best epsilon found using cross-validation: {epsilon}")
print(f"Best F1 on Cross Validation Set: {f1}")
print(f"   (you should see a value epsilon of about 1.38e-18)")
print(f"   (you should see a Best F1 value of 0.615385)")
print(f" Outliers found: {sum(p < epsilon)}")

