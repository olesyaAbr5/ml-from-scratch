import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from functions import train, plotData, plotDecisionBoundary

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', None)

data = np.loadtxt('data.csv', delimiter=',', skiprows=1)
X = data[:, :2]
Y = data[:, 2]
X = np.column_stack([np.ones(X.shape[0]), X])

plotData(data)
plt.title('Точки двух множеств')
plt.show()

np.random.seed(1)
w = np.random.uniform(-1, 1, size=3)
print('Начальные веса:', w)

w, table = train(X, Y, w)
print(table)
print('Эпох:', table['epoch'].max())
print('Итоговые веса:', w)

plotDecisionBoundary(w, data)
plt.show()
