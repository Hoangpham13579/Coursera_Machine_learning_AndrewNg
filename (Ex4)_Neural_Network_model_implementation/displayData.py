import matplotlib.pyplot as plt
import numpy as np


# TODO: Display data labels
def display_data(X, y, fig, ax=None, **plt_kwargs):
    # If ax is None, plt.gca(): get a reference to current existing ax
    if ax is None:
        ax = plt.gca()

    rows = 10; cols = 10
    # Sample randomly 100 images to display from data X
    indexes = np.random.choice(5000, rows*cols)
    count = 0

    # Display 100 random image
    for i in range(0, rows):
        for j in range(0, cols):
            ax = fig.add_subplot(rows, cols, count+1)
            ax.imshow(X[indexes[count]].reshape(20, 20).T, cmap="gray")
            ax.autoscale(False)
            ax.set_axis_off()
            count += 1
    return ax

