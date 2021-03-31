import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from processEmail import email_processing
from processEmail import email_processing_and_index_filtering
import scipy.io
from sklearn import svm
from processEmail import email_features


################## (1) Email Preprocessing & Index considering #########################
# Reading email sample
f = open("data/emailSample1.txt", "r")
content = f.read()

# Email pre-processing
print("Preprocessing emails!!")
content = email_processing(content)
print(content)

# Porter vocabulary list
vocab_df = pd.read_csv("data/vocab.txt", sep="\t", header=None)
vocab_df.columns = ("index", "vocab")

# Filtering index of each word in content from Porter dictionary vocab
print("\nIndex filtering of each word in the content from Porter dictionary list")
indices = email_processing_and_index_filtering(content, vocab_df)
# indices: missing Porter's word index 89 (anywher) in vocab
print(indices)

input("Press enter to continue...")

################### (2) Features extraction #######################
# Each Porter's word in dict is similar to each feature for training SVM model
print("\nCount # of appearance of each Porter's word of content in Porter's Dictionary")

# Count # of appearance of each Porter's word (features) of content in Porter's Dictionary
features = email_features(indices)

print(f"Total number of possible features: {len(features)}")
print(f"Number of non-zero values in X: {np.sum(features == 1)}")
input("Press enter to continue...")


################### (3) Train linear SVM for Spam classification ###############
# Train linear classifier to distinguish between email spam or not
# Load the spam email
data = scipy.io.loadmat("data/spamTrain.mat")

# Train linear SVM model & fit data to model
print("\nTrain linear SVM model (Spam classifiers)")
clf = svm.SVC(kernel="linear", C=0.1)
clf.fit(data["X"], data["y"].ravel())

# Prediction & Accuracy
y = data["y"]
p_train = clf.predict(data["X"])  # 1D array
# accuracy = clf.score(data["X"], data["y"]) (Case 2)
print(f"Training accuracy: {np.mean(p_train == y.ravel())}")
input("Press enter to continue...")


##################### (4) Testing spam classification ##############
# Loading the spam test
print("\nLoading spam test dataset")
spam_test = scipy.io.loadmat("data/spamTest.mat")
Xtest = spam_test["Xtest"]
ytest = spam_test["ytest"]

# Making prediction based on SVM linear classifier model on test set
print("Making prediction on test set")
p_test = clf.predict(Xtest)
print(f"Test accuracy: {np.mean(p_test == ytest.ravel())}")
input("Press enter to continue...")


####################### (5) Top predictors for spam ##################
# Determine the top 15 words which signal the email as spam
# Find the weight of each word in Porter's vocab
# -> The higher the weight of word (1), the more signal the email spam if email includes word (1)
print("\nFinding the weight of each word in Porter's vocab")
vocab_df["weights"] = clf.coef_[0, :]
print(vocab_df.sort_values("weights", axis=0, ascending=False).head(15))
input("Press enter to continue...")


####################### (6) Try classify my own email as spam or not #################
print("\nClassify my own email")

# Read a sample email
print(f"Classify sample email 1")
f = open("data/emailSample1.txt", "r")
content_samp1 = f.read()
# Processing the email
content_samp1 = email_processing(content_samp1)
# Filtering the index for each word & Generate X for training
indices_samp1 = email_processing_and_index_filtering(content_samp1, vocab_df)
X = email_features(indices_samp1)  # (1899, 1)
# Making prediction
pre_samp1 = clf.predict(X.reshape(1, -1))
print(f"Spam classify: {pre_samp1[0]}")
print("Sample email 1 is not the spam email")

# Read a sample email
print("\nClassify sample email 2")
f = open("data/emailSample2.txt", "r")
content_samp2 = f.read()
# Processing the email
content_samp2 = email_processing(content_samp2)
# Filtering the index for each word & Generate X for training
indices_samp2 = email_processing_and_index_filtering(content_samp2, vocab_df)
X = email_features(indices_samp2)  # (1899, 1)
# Making prediction
pre_samp2 = clf.predict(X.reshape(1, -1))
print(f"Spam classify: {pre_samp2[0]}")
print("Sample email 2 is not the spam email")


