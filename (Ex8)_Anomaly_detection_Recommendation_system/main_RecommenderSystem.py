import numpy as np
import scipy.io
import scipy.optimize as opt

from collaborativeFilteringAlgo import cofi_learning_algo
from checkGradient import check_cost_function
from collaborativeFilteringAlgo import normalize_rating


# In this part, we will implement the collaborative filtering algorithm and apply it to the movie rating.
# Purpose: Predict the rating of one user to specific movies based on their already learning movies
# The dataset consists of ratings on a scale 1 to 5. The dataset has n_u = 943 (# of users); n_m = 1682 (# of movies)

####################### (1) Loading movie rating dataset ####################
print("Loading the movie rating dataset")
movies = scipy.io.loadmat("data/ex8_movies.mat")
# Matrix "Y" (num_movies * num_users) ratings y_(i,j) (from 1 to 5)
Y = movies["Y"]
# Matrix "R" (num_movies * num_users) binary where R(i,j)=1 if user j gave a rating to movie i, and R(i,j)=0 otherwise.
R = movies["R"]

movies_param = scipy.io.loadmat("data/ex8_movieParams.mat")
# Matrix "X" (num_movies * num_features) each row correspond to the feature vector x_(i) for the i_th movie
X = movies_param["X"]
# Matrix "theta" (num_movies * num_features) Each row correspond to one parameter vector theta_(j) for the j-th user
Theta = movies_param["Theta"]
num_users = movies_param["num_users"]
num_movies = movies_param["num_movies"]
num_features = movies_param["num_features"]
print(f"Number of users: {num_users}\n"
      f"Number of movies: {num_movies}\n"
      f"Number of features: {num_features}")
print(f"The average rating of movie 1 (Toy story): {np.round(np.mean(Y[0, R[0, :]==1]), 2)} / 5")
input("Program pause, Press enter to continue")


#################### (2) Apply collaborative filtering learning algorithm ################
# Check the correction of cost function (without regularization)"
print("\nChecking the correction of cost function (without regularization)")
# Initial parameters for checking
num_users_check = 4
num_movies_check = 5
num_features_check = 3
X_check = X[:num_movies_check, :num_features_check]
Theta_check = Theta[:num_users_check, :num_features_check]
Y_check = Y[:num_movies_check, :num_users_check]
R_check = R[:num_movies_check, :num_users_check]

# Compute cost for check the correction of cost function (without regularization)
J, _ = cofi_learning_algo(np.hstack((X_check.ravel(order="F"), Theta_check.ravel(order="F"))),
                          Y_check, R_check, num_users_check, num_movies_check, num_features_check, 0)
print(f"Cost's result is: {J}")
print(f"The value of cost should be 22.22")
input("Program pause, Press enter to continue")

# Checking the gradient descent of the cost (without regularization)
print("\nCheck the difference between gradient using formula & numerical gradient (Without regularization)")
check_cost_function(lambda_r=0)
input("Program pause, Press enter to continue")


# Check the correction of cost function (with regularization)
print("\nCheck the correction of cost function with regularization")
J_regu, _ = cofi_learning_algo(np.hstack((X_check.ravel(order="F"), Theta_check.ravel(order="F"))),
                          Y_check, R_check, num_users_check, num_movies_check, num_features_check, 1.5)
print(f"Cost at loaded parameters (lambda = 1.5): {np.round(J_regu, 2)}")
print('\t(this value should be about 31.34)')
input("Program pause, Press enter to continue")

# # Checking the gradient descent of the cost (with regularization)
print("\nCheck the difference between gradient using formula & numerical gradient (With regularization)")
check_cost_function(lambda_r=1.5)
input("Program pause, Press enter to continue")


###################### (3) Learning movie recommendation #########################
# Loading movie's ids dataset
print(f"\nLoading the movie's id dataset")
f = open("data/movie_ids.txt", encoding="latin-1")
content = f.readlines()
movies_list = [" ".join(line.split()[1:]) for line in content]
print(f"The 1st 5 movies in the movie list: {movies_list[:5]}\n"
      f"The total number of movies: {len(movies_list)}")

# Generate some initial rating for a user
my_ratings = np.zeros((1682, 1))
my_ratings[0] = 4
my_ratings[10] = 4
my_ratings[21] = 5
my_ratings[70] = 5
my_ratings[97] = 2
my_ratings[98] = 5
my_ratings[150] = 4
my_ratings[154] = 4
my_ratings[175] = 3
my_ratings[312] = 5
for i, r in enumerate(my_ratings):
    if r > 0:
        print(f"Rate {r[0]} for {movies_list[i]}")
input("Program pause, Press enter to continue")


# Loading movie's rating dataset
print("\nLoading the movies rating dataset")
movies_data = scipy.io.loadmat("data/ex8_movies.mat")
Y = np.hstack((my_ratings, movies_data["Y"]))
R = np.hstack((my_ratings != 0, movies_data["R"]))
print(f"The number of user: {movies_data['Y'].shape[1]}")

# Training collaborative filtering
# Normalize rating dataset Y
print("\nMinimize the cost function to result the optimal theta")
Y_norm, Y_mean = normalize_rating(Y, R)

# Initialize some initial parameters
num_users = Y.shape[1]
num_movies = Y.shape[0]
# (NOTE) Initially, do NOT know detailed features in recommender system which can define them later
num_features = 10
X = np.random.normal(size=(num_movies, num_features))
Theta = np.random.normal(size=(num_users, num_features))
initial_params = np.hstack((X.ravel(order="F"), Theta.ravel(order="F")))

################### Optimize cost to get the optimal theta and X ################
lambda_r = 10
opt_results = opt.minimize(cofi_learning_algo, initial_params,
                           args=(Y, R, num_users, num_movies, num_features, lambda_r),
                           method="L-BFGS-B", jac=True, options={"maxiter": 100})
theta = opt_results["x"]
# Getting the trained theta and X values
X = np.reshape(theta[:num_movies*num_features], newshape=(num_movies, num_features), order='F')
Theta = np.reshape(theta[num_movies*num_features:], newshape=(num_users, num_features), order='F')
input("Program pause, Press enter to continue")


####################### Making the recommendation #####################
print("\nMaking the recommendation of movies for the users (me)")
p = np.dot(X, Theta.T)  # p: (num_movies, num_users)
my_prediction = p[:, 0]
# [::-1] means "1234" -> "4321"
sort_idx = np.argsort(my_prediction)[::-1]

print("Top recommendation for you")
for i in range(10):
    j = sort_idx[i]
    print(f"Predicting rating {my_prediction[j]} for movies {movies_list[j]}")

print("\nOriginal rating provided: ")
for i, r in enumerate(my_ratings):
    if r > 0:
        print('Rated {0} for {1}'.format(int(r[0]), movies_list[i]))

# (NOTE) Collaborative filtering is an unsupervised learning algorithm
