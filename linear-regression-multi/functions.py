import numpy as np

def featureNormalize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)

def computeCostMulti(X, theta, y):
    X = np.column_stack((X, np.ones(X.shape[0])))
    J = 1 / (2 * X.shape[0]) * ((np.dot(X, theta) - y) ** 2).sum()
    return J


def gradientDescentMulti(theta, X, y, alpha):
    X = np.column_stack((X, np.ones(X.shape[0])))
    theta = theta - alpha * (1 / X.shape[0]) * (np.dot(X.T, np.dot(X, theta) - y))
    return theta


def normalEqn(X, y):
    X = np.column_stack((X, np.ones(X.shape[0])))
    theta = np.linalg.pinv(X.T @ X) @ (X.T @ y)
    return theta