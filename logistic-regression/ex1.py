import numpy as np
import matplotlib.pyplot as plt

from functions import plotData, computeCost, gradientDescent, plotDecisionBoundary, predict

data = np.loadtxt('ex2data1.txt', delimiter=',')
X = data[:, :2]
y = data[:, -1]

mean = X.mean(axis=0)
std = X.std(axis=0)
X_norm = (X - mean) / std
X_norm_one = np.column_stack((np.ones(X.shape[0]), X_norm))


plotData(X_norm, y)
plt.show()


X_one = np.column_stack((np.ones(X.shape[0]), X))


num_iters = 10000
alphas = [0.01, 0.05, 1, 1.5, 2, 3]
thetas = {}
for alpha in alphas:
    theta = np.zeros(3)
    compute_costs = []
    for _ in range(num_iters):
        compute_costs.append(computeCost(theta, X_norm_one, y))
        theta = gradientDescent(theta, X_norm_one, y, alpha)
    thetas[alpha] = theta
    plt.plot(range(num_iters), compute_costs, label=f'alpha = {alpha}')
plt.title('Функция стоимости')
plt.xlabel('Итерация')
plt.ylabel('J(theta)')
plt.legend()
plt.show()



theta = thetas[3]
print('theta:', theta)
print('Cost:', computeCost(theta, X_norm_one, y))


plotDecisionBoundary(theta, X_norm, y)
plt.show()


engine = np.array([[69.36458875970939, 97.71869196188608]])
class_predict, p = predict(engine, theta, mean, std)
print('Вероятность неисправности:', p[0])
print('Результат:', 'неисправен' if class_predict == 1 else 'исправен')



# Нелинейный признак x1 * x2
new_column = X_norm[:, 0] * X_norm[:, 1]
new_column_mean = new_column.mean()
new_column_std = new_column.std()
new_column_norm = (new_column - new_column_mean) / new_column_std
X_non_linear_norm = np.column_stack((X_norm, new_column_norm))
X_non_linear_norm_one = np.column_stack((np.ones(X.shape[0]), X_non_linear_norm))

theta_non_linear = np.zeros(4)
compute_costs = []
for _ in range(num_iters):
    compute_costs.append(computeCost(theta_non_linear, X_non_linear_norm_one, y))
    theta_non_linear = gradientDescent(theta_non_linear, X_non_linear_norm_one, y, 0.1)
plt.plot(range(num_iters), compute_costs)
plt.title('Функция стоимости (с признаком x1 * x2)')
plt.xlabel('Итерация')
plt.ylabel('J(theta)')
plt.show()

print('theta с признаком x1 * x2:', theta_non_linear)
print('Cost линейной модели:', computeCost(theta, X_norm_one, y))
print('Cost с признаком x1 * x2:', computeCost(theta_non_linear, X_non_linear_norm_one, y))

x1 = np.linspace(-2, 2, 1000)
x2 = (- theta_non_linear[0] - theta_non_linear[1] * x1) / (theta_non_linear[2] + theta_non_linear[3] * x1)
plotData(X_norm, y)
plt.plot(x1, x2, label='разделяющая кривая')
plt.ylim(-2.5, 2.5)
plt.legend()
plt.show()

