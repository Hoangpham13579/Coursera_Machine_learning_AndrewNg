import seaborn as sns
import numpy as np

from gaussianDistribution import multivariate_gaussian


# MGD: Multivariate Gaussian (Normal) Distribution
def fit_MGD_visualization(X, mu, sigma2):
    # Visualize the dataset X
    ax = sns.scatterplot(X[:, 0], X[:, 1], markers="x", color="b", s=25)
    ax.set(xlabel="Latency (ms)", ylabel="Throughput (mb/s)", title="1st data set")

    # Fitting the MGD result (with mu & sigma of X) to the visualization
    # Generate the an array of index's positions on a grid (Ex: [[[0,0], [0,1],...], [[1,0], [1,1],...], ...])
    x, y = np.mgrid[0:35:0.5, 0:35:0.5]
    pos = np.dstack((x, y))
    z = multivariate_gaussian(pos, mu, sigma2)
    z = z.reshape(x.shape)

    # Plot the MGD fitting line
    ax.contour(x, y, z, levels=10.0**(np.arange(-20, 0, 3)))
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)
    return ax


# "np.mgrid()" return multi-dimensional "meshgrid"
# "np.dstack()" concatenates along the 3rd dimension
# "levels" in ax.contour() helps to increase the width of MGD distribution value

