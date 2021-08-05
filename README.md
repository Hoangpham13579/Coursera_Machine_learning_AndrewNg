# Coursera Machine learning Stanford

- **Arthur Samuel (1959)**

    - ***Machine Learning***: Field of study that gives computers the ability to learn without being explicitly programmed

## Example

- **Database mining**
    - Large datasets from growth of automation/web
    - **E.g.,** Web click data, medical records, biology, engineering
- **Application can't program by hand**
    - **E.g**., Autonomous helicopter, handwriting recognition, most Natural Language Processing (NLP), Computer Vision.
- **Self-customizing programs**
    - **E.g.,** Amazon, Netflix product recommendations
- **Understanding human learning (Brain, real AI)**

## Learning topics

### Machine learning algorithm

- **Supervised learning**
    - Linear regression
    - Logistic regression
    - Neural network
    - Support Vector Machine (SVMs)
- **Unsupervised learning**
    - K-means clustering
    - Principal Component Analysis (PCA)
    - Anomaly detection

### **Special applications / Special topics**

- **Recommender system**
- **Large scale machine learning**

### Advise on building a machine learning system

- ***Bias/Variance, Regularization***
- ***Deciding what to work on next: Evaluations of learning algorithms, learning curves, Error analysis, ceiling analysis***

-------------------------------------------------------------------------

# Linear Regression

### WHEN is Linear regression useful? (NOT DONE)

> **Linear regression** is used to ***predict the real-value output***

- The continuous target features such as "Pricing of the house", "Student's grade",...

## Cost function (Linear regression)

<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/33dcc792-17c7-45b4-b501-a37ab99d342a/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210805%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210805T032148Z&X-Amz-Expires=86400&X-Amz-Signature=60b7ed365d5b166be78c4719d19d0ebfae73ebb10a063b58aa5f195c80731e60&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22" style="width:400px;height:200px;">

**Cost function** equals to 

- 1/2m * ***Square error between ("predicted result from hypothesis function h(x)" & "Value y data")***
- (Theta1, Theta2) are parameters of h(x)
- **Explaining the terms (1/2m)** in J(theta1, theta2)
    - The (1/m) is to "average" the square error over the number of components
    - The term (1/2) exist because, by the rule of thumb, ***2m seems to be more "natural"*** —> Author (Andrew Ng) prefers it & ***When minimizing J, It doesn't matter J or 2J***

## Gradient descent (Linear regression)

- **Objectives**
    - Start with some (theta1, theta2,...) = (0, 0, ...)
    - Keep changing (Theta1, Theta2,...) to reduce cost function J(theta1, theta2) until we hopefully end up at a minimum

![](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/b76c488e-6443-406a-bed7-edcf167a4b45/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210805%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210805T032712Z&X-Amz-Expires=86400&X-Amz-Signature=883b53c26551629c62e30c4d4bf58e15d56bfad820cad3721284082a4b7d5433&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

- **NOTE!!!!!!!!!!!**
    - In "Gradient descent", The ***meanings of J(Theta1, Theta2)'*** is to ***decide the direction of the steps*** (Positive = left & Negative = right)
    - **Learning rate (alpha)** ***decides the size of the step***
    - Global minimum Vs Local minimum

    ![](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/76c57df7-8a56-41f8-8766-f2918fe9083e/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210805%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210805T032736Z&X-Amz-Expires=86400&X-Amz-Signature=0eea7ebaf20685a7e680af5735b6635f772d4c7a34626e0ecc3f4a4fd154ca3c&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

    (**Question!!!**) ***How to solve*** the problem when the ***cost function always converges to a local minimum*** instead of a global minimum?
    
 -------------------------------------------------------------------------
    
 
