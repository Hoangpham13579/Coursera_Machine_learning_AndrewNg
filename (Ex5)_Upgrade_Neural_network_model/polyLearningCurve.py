import numpy as np
import matplotlib.pyplot as plt
from trainLinearReg import trainLinearReg
from polyMappingFeature import poly_features
from polyMappingFeature import feature_normalize
from learningCurve import learningCurve


# TODO: Train theta using poly regression & plot Learning curve based on X_poly
# return: Fit hypothesis line & learning curve
def poly_learning_curve(X, X_poly, y, X_poly_vali, y_vali, lambdas, p):
    # X: (m, n+1); X_poly: (m, p+1); y: (m, 1); X_poly_vali: (m_vali, p+1); y_vali: (m_vali, 1)
    # Train theta based on X polynomial regression
    poly_train_theta = trainLinearReg(X_poly, y, lambdas)  # (p+1, 1)
    fig = plt.figure(figsize=(15, 20))
    m = X.shape[0]

    # Scatter plot data X
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(X[:, 1], y, "x",  color="blue", markersize=7)
    # Plot fit hypothesis line
    x = np.arange(np.min(X[:, 1]) - 15.0, np.max(X[:, 1]) + 25.0, 0.05)
    x_poly_plot = poly_features(x, p)
    x_poly_plot, _, _ = feature_normalize(x_poly_plot)
    x_poly_plot = np.hstack((np.ones((x_poly_plot.shape[0], 1)), x_poly_plot))  # (x.shape[0], p+1)
    ax1.plot(x, np.dot(x_poly_plot, poly_train_theta), "--", color="red", linewidth=3)
    ax1.set_xlabel("Set values of X")
    ax1.set_ylabel("Red line: predicting line; blue: true value (y)")

    # Plot error of train and cross validate data set with respect to number of examples
    ax2 = fig.add_subplot(2, 1, 2)
    error_train_poly, error_vali_poly = learningCurve(X_poly, y, X_poly_vali, y_vali, lambdas)
    ax2.plot(range(1, m+1), error_train_poly, "-", color="blue")
    ax2.plot(range(1, m+1), error_vali_poly, "-", color="red")
    ax2.set_xlabel("Number of examples")
    ax2.set_ylabel("Error")
    ax2.legend(("Poly training error", "Poly validate error"))

    plt.show()

