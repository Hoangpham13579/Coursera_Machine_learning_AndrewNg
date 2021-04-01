import numpy as np
import matplotlib.pyplot as plt
# Open PNG file
from PIL import Image
# Load matlab's file
import scipy.io

from initializeCentroids import initialize_centroids
from KmeansAlgo import kmeans_algo
from findClosestCentroids import find_closest_centroid


# Purpose: we'll use K-means algo to select 16 colors that will be used to represent the compressed image.
# Concretely, we treat every pixel of original image as a data example AND use K-means algo to find 16 colors that are
# the most appropriate cluster (group) of pixel in 3D RGB space. Once we have computed the trained cluster centroids
# on the image, we use these 16 colors to replace colors in image (Means compress image to only 16 colors)

# Load image & show PNG image
fname = "data/bird_small.png"
plt.imshow(Image.open(fname))
plt.axis("off")
plt.show()

# Load image array file of "bird_small.png"
bird_mat = scipy.io.loadmat("data/bird_small.mat")
A = bird_mat["A"]

# Scaling and Reducing dimension to 2D array
A = np.divide(A, 255)  # Result between 0-1
A = np.reshape(A, (A.shape[0]*A.shape[1], A.shape[2]))
input("Pause program, Press enter to continue")


############# Run K-mean to compress colors of image to 16 colors #############
print("Run K-means clustering algo with K = 16 ")
# Set Initial parameters
K = 16
max_iter = 10
initial_centroids = initialize_centroids(A, K)

# Fitting Data to K-means model -> trained centroids
centroids, _ = kmeans_algo(A, initial_centroids, max_iter, False)  # (K, A.shape[1])

# Apply K-mean model to compress image
# "idx": index of each color closet to each data example
idx = find_closest_centroid(A, centroids)
idx = idx.astype(int)  # (A.shape[0], 1)

# Recover image with only 16 colors
A_recover = np.zeros((A.shape[0], A.shape[1]))
# Apply closest color to each data example
for k in range(A_recover.shape[0]):
    A_recover[k, :] = centroids[idx[k]-1, :]
# Reshape recover array to an image array (3D arr)
A_recovered = A_recover.reshape((bird_mat["A"].shape[0], bird_mat["A"].shape[1], bird_mat["A"].shape[2]))

# Plotting original image and recovered image for comparing
plt.subplot(2, 1, 1)
plt.axis("off")
plt.title("Original")
plt.imshow(Image.open(fname))

plt.subplot(2, 1, 2)
plt.title("Recover")
plt.axis("off")
plt.imshow(A_recovered)
plt.show()
input("Pause program, Press enter to continue")


############## K-means with multiple value of K ###############
print("\nRun K-means clustering algorithm with multiple value of K")
# Data input "A" is already pre-precessed above
K_vals = [2, 8, 16, 24, 32]
max_iter = 10

# Run K-means algo to different value of K
img_compressed = []
for iK, K in enumerate(K_vals):
    print(f"\nRun K-mean algo with K = {K}")

    # Run K-means algo
    initial_centroids = initialize_centroids(A, K)
    centroids, _ = kmeans_algo(A, initial_centroids, max_iter, False)
    # Image compression
    idx = find_closest_centroid(A, centroids)
    idx = idx.astype(int)
    # Recover the image with K colors
    A_recover = np.zeros((A.shape[0], A.shape[1]))
    for k in range(A_recover.shape[0]):
        A_recover[k, :] = centroids[idx[k]-1, :]
    A_recovered = A_recover.reshape((bird_mat["A"].shape[0], bird_mat["A"].shape[1], bird_mat["A"].shape[2]))
    # Add recovered image to list
    img_compressed.append(A_recovered)

input("Pause program, Press enter to continue")


# Multiple plot for different recovered images
fig = plt.figure(figsize=(7, 9))
nrows = 3
ncols = 2
ax = fig.add_subplot(nrows, ncols, 1)
ax.axis("off")
ax.set_title("Original")
ax.imshow(bird_mat["A"])

for i in range(len(img_compressed)):
    ax = fig.add_subplot(nrows, ncols, i+2)
    ax.imshow(img_compressed[i])
    ax.axis("off")
    ax.set_title(f"K = {K_vals[i]}")
plt.show()

