import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns

from featureNormalize import feature_normalize
from PCA_algo import pca
from projectData import project_data
from recoverData import recover_data


# In this exercise, we'll use principal component analysis (PCA) to perform dimensionality reduction.
########### Implement PCA on 2D dataset ####################
print(f"Plotting data example 1")
data1 = scipy.io.loadmat("data/ex7data1.mat")
sns.scatterplot(data1["X"][:, 0], data1["X"][:, 1])
plt.show()

input("Pause program, Press enter to continue")

# Implement PCA (Reduce from 2D to 1D -> k=1)
# (1st compute the covariance matrix of dataset. 2nd use svd function to compute eigenvectors U1, U2,...
# Un which are corresponding to principal component (main component) of variation (thành phần) dataset)
X_norm, mu, std = feature_normalize(data1["X"])
U, S = pca(X_norm)

# Visualize the dimensional reduction data resulted by PCA
# Visualize original data
print(f"\nVisualize dimensional reduction data")
data1 = scipy.io.loadmat("data/ex7data1.mat")
ax = sns.scatterplot(data1["X"][:, 0], data1["X"][:, 1])
# Compute & visualize compressed data (Z)
p1 = mu  # (1, 2)
p2 = mu + 1.5 * S[0] * U[:, 0].T  # (1, 2)
ax.plot((p1[0], p2[0]), (p1[1], p2[1]), "k")

p1 = mu
p2 = mu + 1.5 * S[1] * U[:, 1].T
ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k')
plt.title('Computed eigenvectors of the dataset')
plt.show()
input("Pause program, Press enter to continue")


############ Dimensionality reduction with PCA ##############
print(f"\nReduce dimensionality of data from n-dim to K-dim")
# Checking correctness of reduction
K = 1
Z = project_data(X_norm, U, K)  # (m, K)
print(f"Projection of the 1st example: {Z[0]}")
print(f"This value should be 1.481274")

# Reconstructing compressed data approximately to original data
print(f"\nReconstruct compressed data to approximately original data")
X_rec = recover_data(Z, U, K)
print(f"Approximation of 1st example: {X_rec[0, 0]}, {X_rec[0, 1]}")
print(f"The result should be -1.047419 -1.047419")

# Visualize the projection
ax = sns.scatterplot(X_norm[:, 0], X_norm[:, 1], s=50)
ax = sns.scatterplot(X_rec[:, 0], X_rec[:, 1], s=50, color="r")
for i in range(X_norm.shape[0]):
    p1 = X_norm[i, :]
    p2 = X_rec[i, :]
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "k--")
plt.title("Normalized and projected data after applying PCA")
plt.show()
input("Pause program, Press enter to continue")


################## Face image dataset ######################
# Loading faces data
faces_data = scipy.io.loadmat("data/ex7faces.mat")
# Visualize the first 100 pic of faces
print("\nVisualize the first 100 pictures of faces")
rows = 7
cols = 7
count = 0
fig = plt.figure(figsize=(5, 5))
for row in range(rows):
    for col in range(cols):
        ax = fig.add_subplot(rows, cols, count+1)
        ax.imshow(faces_data["X"][count].reshape(32, 32).T, cmap="gray")
        ax.axis("off")
        count += 1
plt.show()
input("Pause program, Press enter to continue")


# Normalize data X & compute svd() function
X_norm, mu, std = feature_normalize(faces_data["X"])  # X_norm: (5000, 1024) (5000 pic)
U, S = pca(X_norm)

# Apply PCA to data with K = 100
K = 100
Z = project_data(X_norm, U, K)  # (m, K)
# Visualize the first 100 dimensional reduction pic of face
print("\nDrawing 1st 100 pic of face with dimensional reduction to K = 100")
rows = 7
cols = 7
count = 0
fig = plt.figure(figsize=(5, 5))
for row in range(rows):
    for col in range(cols):
        ax = fig.add_subplot(rows, cols, count+1)
        ax.imshow(Z[count, :].reshape(10, 10).T, cmap="gray")
        ax.axis("off")
        count += 1
plt.show()
input("Pause program, Press enter to continue")


# Reconstructing from compressed image to original image
X_rec = recover_data(Z, U, K)
print("\nPlot the reconstructed image from compressed image (From 10*10 to 32*32)")
rows = 7
cols = 7
count = 0
fig = plt.figure(figsize=(5, 5))
for row in range(rows):
    for col in range(cols):
        ax = fig.add_subplot(rows, cols, count+1)
        ax.imshow(X_rec[count, :].reshape(32, 32).T, cmap="gray")
        ax.axis("off")
        count += 1
plt.show()

