import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from findClosestCentroids import find_closest_centroid
from computeCentroidMean import compute_centroid_mean


# TODO: Plot data point X with current and previous centroids
# Output: a figure that contains data X, current & previous centroids (which are connected by dash line)
def plot_kmeans(X, centroids, previous, idx, K, axes):
    for i, ax in enumerate(axes):
        # "hue": grouping data with different color & "palette": Method choosing color for each data group
        # "ax": using the pre-existing ax for the plot (same as plt.gca())
        # Plot data X and centroids point
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=idx.ravel(), legend=False, palette=['r', 'g', 'b'], ax=ax)
        sns.scatterplot(x=centroids[:, 0], y=centroids[:, 1], marker="X", color="k", legend=False, ax=ax, s=100)

        # Going through each group of data (cluster) (2D in this case)
        for k in range(centroids.shape[0]):
            # Sketch dash line connecting from previous centroid to current centroids
            ax.plot([previous[k, 0], centroids[k, 0]], [previous[k, 1], centroids[k, 1]], "k--")


# TODO: Perform Kmeans clustering algorithm
# Input: X: data, "initial_centroids": set of initial centroids
# "max_iter": # of iterations
# Return the trained centroids AND idx (set of m params as the closest index of centroid)
def kmeans_algo(X, initial_centroids, max_iter, plot_progress):

    # Plotting progress only valid for 3 colors (K=3)
    if plot_progress:
        # Setting positions for each plot
        ncols = 3
        nrows = int(max_iter/ncols)
        if (max_iter % ncols) > 0:
            nrows = nrows + 1

        # Create number of "fig" & associated "axes" relating to each number of iterations
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, nrows*8))
        # Create list of index for each grid (in an array of grids)
        ax_tupple = list(np.ndindex(nrows, ncols))
        # Dealing with the fig not containing iteration of K-means algorithm
        for ax in ax_tupple[max_iter:]:
            # Remove the unused ax off displaying
            axes[ax].set_axis_off()

        # Choose needed grid's index
        ax_tupple = ax_tupple[:max_iter]

    # At first "centroids" & "Initial centroids" are the same
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids


    # Perform k-means algorithm
    for iter in range(max_iter):
        # List index of closest centroids of each data point in X
        idx = find_closest_centroid(X, centroids)

        # Plotting progress only valid for 3 colors (K=3)
        if plot_progress:
            # List of "axes" for each grid from the current plotting grid
            plot_axes = [axes[ax_idx] for ax_idx in ax_tupple[iter:]]
            # Set title for each iteration
            axes[ax_tupple[iter]].set_title(f"K-means iteration {iter+1}/{max_iter}")
            # Plot the K-means
            plot_kmeans(X, centroids, previous_centroids, idx, K, plot_axes)
            previous_centroids = centroids
            previous_ax = plt.gca()
        else:
            print(f"K-means iteration {iter+1}/{max_iter}")

        # Update centroids for each iteration
        centroids = compute_centroid_mean(X, idx, K)

    if plot_progress:
        plt.show()

    return centroids, idx

