import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def f(z):
    return 1 if z > 0 else 0


def train(X, Y, w):
    epoch = 0
    error = 1
    rows = []
    while error != 0:
        epoch += 1
        error = 0
        for i in range(X.shape[0]):
            z = np.dot(X[i], w)
            y_pred = f(z)
            if Y[i] != y_pred:
                error += 1
                if y_pred == 0:
                    w += X[i]
                else:
                    w -= X[i]

            rows.append([epoch, *X[i], z, y_pred, Y[i], *w.copy()])

    table = pd.DataFrame(rows, columns=['epoch', 'x0', 'x1', 'x2', 'z', 'y_pred', 'Y', 'w0', 'w1', 'w2'])
    return w, table


def plotData(data):
    mask0 = data[data[:, 2] == 0]
    mask1 = data[data[:, 2] == 1]
    plt.scatter(mask0[:, 0], mask0[:, 1], marker='o', color='blue', label='y = 0')
    plt.scatter(mask1[:, 0], mask1[:, 1], marker='x', color='red', label='y = 1')
    plt.xlim(-2, 3)
    plt.ylim(-2, 3)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid()
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.legend()


def plotDecisionBoundary(w, data):
    plotData(data)
    x1 = np.linspace(-2, 3, 100)
    x2 = (-w[0] - w[1] * x1) / w[2]
    plt.plot(x1, x2, label='разделяющая прямая')
    plt.title('Разделяющая прямая персептрона')
    plt.legend()
