import numpy as np
import scipy.io
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
from plotBoundary import plot_boundary
from gaussianKernel import gaussian_kernels
from plotBoundary import plot_gaussian_boundary
from gaussianKernel import gaussian_kernel_matrix
from SVM_kernels_param import svm_kernel_param


# Loading data1
data1 = scipy.io.loadmat("data/ex6data1.mat")

# Visualize data1
ax = sns.scatterplot(x=data1["X"][:, 0], y=data1["X"][:, 1], hue=data1["y"].ravel(), legend=False)
ax.set_title(label="Example 1 dataset")
plt.show()

input("Press Enter to continue...")

##################### (1) Visualize dataset 1 with different value of C #################

print(f"\nSketch decision boundary for different value of C")
# Sketch decision boundary for different value of C
c_vals = [1.0, 100.0]
fig = plt.figure(figsize=(15, 15))

for i, c in enumerate(c_vals):
    # Construct SVC model & fit training data1 to model
    clf = svm.SVC(kernel="linear", C=c)
    clf.fit(data1["X"], data1["y"].ravel())

    # Plot decision boundary with respect to each value of C (linear)
    fig.add_subplot(2, 1, i+1)
    plot_boundary(data1["X"], data1["y"], clf, f"Decision boundary with c={c}")

plt.show()
input("Press Enter to continue...")


################### (2) Implementing SVM with Gaussian kernel (using "f" value) ###########
print(f"\nChecking result Gaussian kernel computation")
# Checking result Gaussian kernel computation
x1 = np.array((1, 2, 1))
x2 = np.array((0, 4, -1))
sigma = 2
sim = gaussian_kernels(x1, x2, sigma)
print(f"Gaussian Kernel between x1 and x2, sigma = {sigma} : {sim}")
print(f"This value should be about 0.324652\n")
input("Press Enter to continue...")


################ (3) Visualize dataset 2 & Plot their decision boundary (non-linear) ##########2######
data2 = scipy.io.loadmat("data/ex6data2.mat")

# Visualize data2
print(f"\nVisualize training dataset 2")
ax = sns.scatterplot(x=data2["X"][:, 0], y=data2["X"][:, 1], hue=data2["y"].ravel(), legend=False)
ax.set_title(label="Example 2 dataset")
plt.show()

input("Press Enter to continue...")

# Plot decision boundary for SVM model apply kernel tricks
# Construct SVC model & fit training data1 to model
clf = svm.SVC(kernel="precomputed", C=1.0)
# "gram": Gaussian matrix
gram = gaussian_kernel_matrix(data2["X"], data2["X"])
clf.fit(gram, data2["y"].ravel())

# Plot decision boundary with c = 1.0
print("\nPlot Gaussian decision boundary for SVM model apply kernel tricks")
plot_gaussian_boundary(data2["X"], data2["y"], clf, f"Gaussian decision boundary with c=1.0, sigma=0.1")

input("Press Enter to continue...")


################ (4) Visualize dataset 3 & Plot their decision boundary (non-linear) ##########2######
data3 = scipy.io.loadmat("data/ex6data3.mat")

# Visualize data3
print(f"\nVisualize training dataset 3")
ax = sns.scatterplot(x=data3["X"][:, 0], y=data3["X"][:, 1], hue=data3["y"].ravel(), legend=False)
ax.set_title(label="Example 3 dataset")
plt.show()
input("Press Enter to continue...")


# Plot decision boundary for SVM model apply kernel tricks
print("\nPlot Gaussian decision boundary for SVM model apply kernel tricks")

# Finding the best value of C and Sigma
bestC, bestSigma = svm_kernel_param(data3["X"], data3["y"].ravel(), data3["Xval"], data3["yval"].ravel())
print("Optimal value of C is: ", bestC)
print("Optimal value of sigma is: ", bestSigma)

# Construct SVC model & fit training data1 to model
clf = svm.SVC(kernel="precomputed", C=bestC)

# "gram": Gaussian matrix
gram = gaussian_kernel_matrix(data3["X"], data3["X"])
clf.fit(gram, data3["y"].ravel())

# Plot decision boundary with c = 1.0
plot_gaussian_boundary(data3["X"], data3["y"].ravel(), clf, f"Gaussian decision boundary with c=bestC")

