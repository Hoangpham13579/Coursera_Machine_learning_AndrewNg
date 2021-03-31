import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from gaussianKernel import gaussian_kernel_matrix


# TODO: Plotting SVM boundary for training data set (LINEAR line)
def plot_boundary(X, y, model, title):
    ax = sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y.ravel(), legend=False)
    ax.set_title(label=title)
    # Plot current "ax" on current figure (fig)
    plt.gca()
    Xlim = ax.get_xlim()  # return (min, max) of X axis
    ylim = ax.get_ylim()

    # Create grid to evaluate model
    xx = np.linspace(Xlim[0], Xlim[1], num=30)  # Generate 30 samples from Xlim[0] to Xlim[1]
    yy = np.linspace(ylim[0], ylim[1], num=30)
    YY, XX = np.meshgrid(yy, xx)  # Return coordinate matrix XX shape (len(xx), len(yy))
    xy = np.vstack([XX.ravel(), YY.ravel()]).T  # xy: (30*30, 2)

    Z = model.decision_function(xy).reshape(XX.shape)
    # "ax.contour()" draw contour (đường viền) depends on height as value of Z
    a = ax.contour(XX, YY, Z, colors="g", levels=[0], linestyles="--")


# TODO: Sketching SVM decision boundary for Gaussian kernels (NON-LINEAR line)
def plot_gaussian_boundary(X, y, model, title="SVM decision boundary for Gaussian kernels"):
    ax = sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y.ravel(), legend=False)
    ax.set_title(label=title)
    # Plot current "ax" on current figure (fig)
    plt.gca()
    Xlim = ax.get_xlim()  # return (min, max) of X axis
    ylim = ax.get_ylim()

    # Create grid to evaluate model
    xx = np.linspace(Xlim[0], Xlim[1], num=30)  # Generate 30 samples from Xlim[0] to Xlim[1]
    yy = np.linspace(ylim[0], ylim[1], num=30)
    YY, XX = np.meshgrid(yy, xx)  # Return coordinate matrix XX shape (len(xx), len(yy))
    xy = np.vstack([XX.ravel(), YY.ravel()]).T  # xy: (30*30, 2)

    # Generate decision array
    gauss_matrx = gaussian_kernel_matrix(xy, X)
    Z = model.decision_function(gauss_matrx).reshape(XX.shape)

    # Drawing decision boundary (contour)
    ax.contour(XX, YY, Z, colors="g", levels=[0.5], linestyles="--")
    plt.show()


