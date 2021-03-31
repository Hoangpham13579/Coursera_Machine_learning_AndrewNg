import numpy as np
# Load matlab's file
import scipy.io
# Build-in K-means clustering model of sklearn lib
from sklearn.cluster import KMeans
from findClosestCentroids import find_closest_centroid
from computeCentroidMean import compute_centroid_mean
from KmeansAlgo import kmeans_algo

############### (1) Find the closest centroids ####################
print(f"Find the closest centroids")

# Loading the example dataset
data2 = scipy.io.loadmat("ex7data2.mat")

# Find the closest centroid with with initial K=3
K = 3
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
idx = find_closest_centroid(data2["X"], initial_centroids)
print(f"Closest centroids of the 1st 3 examples: {idx[:3].T}")
print(f"The closet centroids should be 1, 3, 2 respectively")
input("Pause program, Press enter to continue")


############## (2) Compute the centroids means ################
print(f"\nCompute means of data point closed to each associated centroid")
centroids = compute_centroid_mean(data2["X"], idx, K)
print(f"Centroids computed after initial finding of closest centroid:\n {centroids}\n")
print(f"(the centroids should be\n [ 2.428301 3.157924 ]\n[ 5.813503 2.633656 ]\n[ 7.119387 3.616684 ]")
input("Pause program, Press enter to continue")


################ (3) K-means clustering ######################
print(f"\nRunning K-means clustering on example dataset")

# Initial setting
K = 3
max_iter = 10
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# Run K-means clustering algorithm. "true" values at last parameter indicate plotting the progress of K-means
centroids, idx = kmeans_algo(data2["X"], initial_centroids, max_iter, True)
input("Pause program, Press enter to continue")


############# (4) K-means clustering using sklearn lib ###############
# from sklearn.cluster import KMeans
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
# Create KMeans model
kmeans_model = KMeans(n_clusters=K, max_iter=max_iter, init=initial_centroids)
# Fit data to the model
kmeans_model.fit(data2["X"])

# Return trained centroids
kmeans_model_centroids = kmeans_model.cluster_centers_
# Comparing centroids of 2 ways
print(f"Trained cluster centroid of sklearn model:\n"
      f"{kmeans_model_centroids[0]}\n"
      f"{kmeans_model_centroids[1]}\n"
      f"{kmeans_model_centroids[2]}\n")

print(f"Trained cluster centroid of manual generation:\n"
      f"{centroids[0]}\n"
      f"{centroids[1]}\n"
      f"{centroids[2]}\n")

input("Pause program, Press enter to continue")


