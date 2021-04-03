import numpy as np

from collaborativeFilteringAlgo import cofi_learning_algo


# TODO: Compute explicitly the gradient (derivative) of cost function by NUMERICAL DIFFERENTIATION algorithm
# Return: the gradient of cost computed by numerical differentiation
# Purpose: Use this to compare with gradient function of cost for checking the correction
def compute_numerical_gradient(theta, Y, R, num_users, num_movies, num_features, lambda_r):
    # Initialize parameters
    e = 0.0001
    num_grad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)

    # Compute numerical gradient (derivative) of cost
    for p in range(len(theta)):
        perturb[p] = e
        loss1, _ = cofi_learning_algo(theta-perturb, Y, R, num_users, num_movies, num_features, lambda_r)
        loss2, _ = cofi_learning_algo(theta+perturb, Y, R, num_users, num_movies, num_features, lambda_r)
        num_grad[p] = (loss2 - loss1)/(2*e)
        perturb[p] = 0
    return num_grad


# TODO: Create an own test to compare the result of gradient's function of cost & explicitly numerical gradient of cost
# Return: zip(grad, num_grad) for comparing & "diff" as difference checking technique between them
def check_cost_function(lambda_r=0):
    #  num_movies=4, num_users=5, num_features=3
    # X_t & Theta_t only used to generate Y and R
    X_t = np.random.uniform(0, 1, (4, 3))
    Theta_t = np.random.uniform(0, 1, (5, 3))

    # Generate suitable array Y and R for collaborative filtering algorithm
    Y = np.dot(X_t, Theta_t.T)  # Y: (4, 5)
    # Choose random index and set value of them to 0
    Y[np.random.uniform(0, 1, Y.shape) > 0.5] = 0
    # The rating is from 1 to 5 stars
    R = np.zeros(Y.shape)
    # If Y = 0 -> user not rate this movie -> R[Y==0] = 0 AND vice versa
    R[Y != 0] = 1

    # Initialize needed params
    X = np.random.normal(size=X_t.shape)
    Theta = np.random.normal(size=Theta_t.shape)
    num_users = Theta.shape[0]
    num_movies = X.shape[0]
    num_features = X.shape[1]


    # Compute gradient for comparing
    params = np.hstack((X.ravel(order="F"), Theta.ravel(order="F")))
    cost, grad = cofi_learning_algo(params, Y, R, num_users, num_movies, num_features, lambda_r)
    num_grad = compute_numerical_gradient(params, Y, R, num_users, num_movies, num_features, lambda_r)

    print("The columns should be similar")
    for i, j in zip(grad, num_grad):
        print(i, j)

    # The way of comparing difference between 2 columns
    diff = np.linalg.norm(num_grad-grad) / np.linalg.norm(num_grad+grad)
    print(f"If your cost function implementation is correct, then the relative difference will "
          f"be small (less than 1e-9). Relative Difference: {diff}")
    return None



# "np.random.uniform()": create array drawn from a Uniform Distribution

# Comparing the difference between 2 columns ("diff": relative difference)
# --> diff = np.linalg.norm(col1-col2) / np.linalg.norm(col1+col2)
# "diff" should be smaller than (1e-9)

