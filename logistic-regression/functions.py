import numpy as np
import matplotlib.pyplot as plt


def plotData(X, y):
    data = np.column_stack((X, y))
    X_not_work = data[data[:, -1] == 1][:, :2]
    X_work = data[data[:, -1] == 0][:, :2]
    plt.scatter(X_not_work[:, 0], X_not_work[:, 1], marker='o', c='red', label='Not work')
    plt.scatter(X_work[:, 0], X_work[:, 1], marker='+', c='blue', label='Work')
    plt.title('Визуализация данных')
    plt.xlabel('Вибрация (норм.)')
    plt.ylabel('Неравномерность вращения двигателя (норм.)')
    plt.legend()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def computeCost(theta, X, y):
    m = X.shape[0]
    z = np.dot(X, theta)
    h = sigmoid(z)
    loss = - 1 / m * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return loss


def gradientDescent(theta, X, y, alpha):
    h = sigmoid(np.dot(X, theta))
    gradient = 1 / X.shape[0] * np.dot(X.T, (h - y))
    new_theta = theta - alpha * gradient
    return new_theta


def plotDecisionBoundary(theta, X, y):
    x1 = np.linspace(-2, 2, 1000)
    x2 = (- theta[0] - theta[1] * x1) / theta[2]
    plotData(X, y)
    plt.plot(x1, x2, label='разделяющая прямая')
    plt.legend()


def predict(X, theta, mean, std):
    X_norm = (X - mean) / std
    X_norm_one = np.column_stack(([1], X_norm))
    h = sigmoid(np.dot(X_norm_one, theta))
    if h > 0.5:
        return 1, h
    else:
        return 0, h
