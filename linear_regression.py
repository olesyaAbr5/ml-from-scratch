"""
Функции для линейной регрессия одной переменной.

Задача - предсказать прибыль автосервиса (СТО) по числу автомобилей в
городе. Ниже даны три эквивалентные реализации функции стоимости и
градиентного спуска (циклом, поэлементно через numpy и полностью
векторно) - они используются для взаимной проверки друг друга, - а
также функция для оценки качества модели и вспомогательные функции для
визуализации.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# ============================
# Разбиение на выборки
# ============================

def trainTest(data):
    """Делит датафрейм на обучающую (70%) и тестовую (30%) выборки."""
    train, test = train_test_split(data, test_size=0.3, random_state=20)
    return train, test


# ============================
# Функции стоимости
# ============================

def computeCost1(X, y, theta):
    """Считает J(theta) циклом по объектам выборки."""
    total = 0
    N = len(X)
    for i in range(N):
        total += ((X[i] * theta[1] + theta[0]) - y[i]) ** 2
    total = total / (2 * N)
    return total


def computeCost2(X, y, theta):
    """Считает J(theta) поэлементно, с помощью операций numpy."""
    N = len(X)
    X = np.array(X)
    y = np.array(y)
    theta = np.array(theta)
    total = sum(((X * theta[1] + theta[0]) - y) ** 2) / (2 * N)
    return total


def computeCost3(X, y, theta):
    """Считает J(theta) в матрично-векторной форме."""
    N = len(X)
    X = np.array(X)
    y = np.array(y)
    one = np.ones(N)
    X = np.column_stack([one, X])
    return sum((y - np.dot(X, theta)) ** 2) / (2 * N)


# ============================
# Градиентный спуск
# ============================

def gradientDescent1(X, y, theta, alpha):
    """Один шаг градиентного спуска, реализованный циклом."""
    N = len(X)
    diff_theta0 = 0
    diff_theta1 = 0
    for i in range(N):
        diff_theta0 += 2 * (theta[0] + theta[1] * X[i] - y[i])
        diff_theta1 += 2 * (theta[0] + theta[1] * X[i] - y[i]) * X[i]
    new_theta = [theta[0] - alpha * diff_theta0 / N, theta[1] - alpha * diff_theta1 / N]
    return new_theta


def gradientDescent2(X, y, theta, alpha):
    """Один шаг градиентного спуска, поэлементно через numpy."""
    N = len(X)
    X = np.array(X)
    y = np.array(y)
    theta = np.array(theta)
    diff_theta0 = 2 * (theta[0] + theta[1] * X - y)
    diff_theta1 = 2 * (theta[0] + theta[1] * X - y) * X
    new_theta = [theta[0] - alpha * sum(diff_theta0) / N, theta[1] - alpha * sum(diff_theta1) / N]
    return new_theta


def gradientDescent3(X, y, theta, alpha):
    """Один шаг градиентного спуска в матрично-векторной форме."""
    N = len(X)
    X = np.array(X)
    one = np.ones(N)
    X = np.column_stack([one, X])
    y = np.array(y)
    theta = np.array(theta)

    errors = np.dot(X, theta) - y
    grad = 2 * np.dot(X.T, errors)
    new_theta = theta - alpha * grad / N
    return new_theta


# ============================
# Оценка качества
# ============================

def predict(test_x, test_y, theta):
    """Возвращает J(theta) на переданной выборке (используется как оценка качества)."""
    return computeCost1(test_x, test_y, theta)


# ============================
# Визуализация
# ============================

def plot_data(x, y):
    """Строит диаграмму рассеяния по сырым данным."""
    plt.scatter(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Визуализация данных')
    plt.show()


def plot_model(x, y, theta):
    """Рисует данные и обученную линейную модель поверх них."""
    x = np.array(x)
    y = np.array(y)
    plt.scatter(x, y)
    one = np.ones(len(x))

    def f(x):
        x = np.column_stack([one, x])
        return np.dot(x, theta)

    plt.plot(x, f(x), color='red')
    plt.xlabel('Число автомобилей, тыс')
    plt.ylabel('Прибыль СТО')
    plt.title('Визуализация данных')
    plt.show()


def plot_theta(cost_history1, cost_history2, cost_history3):
    """Строит график сходимости J(theta) для всех трёх реализаций."""
    plt.plot(range(len(cost_history1)), cost_history1, color='blue', label='циклы')
    plt.plot(range(len(cost_history2)), cost_history2, color='orange', label='поэлементно', linestyle='--')
    plt.plot(range(len(cost_history3)), cost_history3, color='green', label='векторный', linestyle=':')
    plt.legend()
    plt.xlabel('Итерации')
    plt.ylabel('J(θ)')
    plt.title('Сходимость градиентного спуска')
    plt.show()
