# from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Normalize feature matrix X
def normalize(X):
    _min = np.min(X, axis=0)
    _max = np.max(X, axis=0)
    _range = _max - _min
    norm_X = 1 - (maxs - X)/_range
    return norm_X

# Logistic(sigmoid) function
def logistic_function(theta, X):
    return 1.0/(1 + np.exp(-np.dot(X, theta.T)))

# Logistic gradient function
def logistic_gradient(theta, X, y):
    first = logistic_function(theta, X) - y.reshape(X.shape[0], -1)
    final = np.dot(first.T, X)
    return final

# Cost function
def cost_function(theta, X, y):
    log_func = logistic_function(theta, X)
    y = np.squeeze(y)
    step1 = y * np.log(log_func)
    step2 = (1 - y) * np.log(1.0 - log_func)
    final = - step1 - step2
    return np.mean(final)

# Gradient descent function
def gradient_descent(theta, X, y, lr=0.05, converge_change=1e-4):
    cost = cost_function(theta, X, y)
    change_cost = 1
    num_iter = 1

    while(change_cost > converge_change):
        old_cost = cost
        theta = theta - (lr * logistic_gradient(theta, X, y))
        cost = cost_function(theta, X, y)
        change_cost = cost - old_cost
        num_iter += 1
    
    return theta, num_iter

# Predict function
def predict_values(theta, X):
    predict_prob = logistic_function(theta, X)
    predict_value = np.where(predict_prob >= 0.5, 1, 0)
    return np.squeeze(predict_value)

if __name__ == "__main__":
    # Load data from csv file
    df = pd.read_csv("mobile_price/train.csv")
    # print(df.shape)

    # Get features from the data
    X = df[df.columns[:12]].values  # Get 12 features for input
    Y = df[df.columns[-1]].values   # Get the last feature for output
    
    np.random.seed(1)
    p = np.random.permutation(len(X))

    # 60% train - 40% test
    alpha = 0.6
    x_train = X[p[:int(len(X)*alpha)]].copy()
    y_train = Y[p[:int(len(X)*alpha)]].copy()
    x_test = X[p[int(len(X)*alpha):]].copy()
    y_test = Y[p[int(len(X)*alpha):]].copy()

    # Initial beta values 
    theta = np.matrix(np.zeros(x_train.shape[1]))
    # print(theta.shape)
    # Theta values after running gradient descent
    theta, num_iter = gradient_descent(theta, x_train, y_train)
    # Estimated beta values and number of iterations 
    print("Estimated regression coefficients:", theta) 
    print("Number of iterations:", num_iter) 

    y_predict = predict_values(theta, x_test)

    num_correct = len(y_predict[y_predict!=y_test])
    num_total = len(y_predict)
    print("Wrong predict = {}/{}".format(num_correct, num_total))
    print("Logistic Regression model accuracy: {}%".format(round(100*num_correct/num_total, 2)))
