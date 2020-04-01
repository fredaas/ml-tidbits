#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams

rcParams["font.family"] = "monospace"

def mse(w, x, y_pred, y):
    m = len(y)
    return (1 / (2 * m)) * np.sum(np.square(y_pred - y))

def gradient_descent(w, x, y_pred, y, alpha):
    m = len(y)
    return w - alpha * (1 / m) * (x.T.dot(y_pred - y))

def plot_regression(w, x, y, alpha, iterations=200):
    x_b = np.c_[np.ones((len(x) ,1)), x]

    _, ax = plt.subplots(figsize=(5, 5))

    y_pred = 0
    for i in range(iterations):
        y_pred = np.dot(x_b, w)
        w = gradient_descent(w, x_b, y_pred, y, alpha)
        if i % 10 == 0:
            ax.plot(x, y_pred, "r-", linewidth=.25, antialiased=True)

    error = mse(w, x_b, y_pred, y)

    print("Error: {:.4}".format(error))
    print("Weights: {:.4} {:.4}".format(*w[0], *w[1]))

    ax.plot(x, y, "x")

    ax.set_ylabel("y")
    ax.set_xlabel("x")
    ax.legend([ "h(x)" ])
    ax.set_ylim([0, 12])

    plt.title("Linear Regression (Numeric)")
    plt.grid(linestyle="dotted")
    plt.tight_layout()

    plt.savefig("regression.pdf", format="pdf")

def plot_error(w, x, y, learning_rate, iterations=200):
    x_b = np.c_[np.ones((len(x), 1)), x]
    error_history = np.zeros(iterations)

    _, ax = plt.subplots(figsize=(5, 5))

    for alpha in learning_rate:
        for i in range(iterations):
            y_pred = np.dot(x_b, w)
            w = gradient_descent(w, x_b, y_pred, y, alpha)
            error_history[i] = mse(w, x_b, y_pred, y)
        ax.plot(range(iterations), error_history, "x")

    ax.legend([ "alpha = {:.2}".format(x) for x in learning_rate])
    ax.set_ylabel("J(w)")
    ax.set_xlabel("Iterations")

    plt.title("Numeric Error")
    plt.grid(linestyle="dotted")
    plt.tight_layout()
    plt.savefig("error.pdf", format="pdf")

def plot_data(x, y):
    _, ax = plt.subplots(figsize=(5, 5))

    ax.plot(x, y, "x")

    ax.set_ylabel("y")
    ax.set_xlabel("x")
    ax.set_ylim([0, 12])

    plt.title("Data")
    plt.grid(linestyle="dotted")
    plt.tight_layout()

    plt.savefig("data.pdf", format="pdf")

if __name__ == "__main__":
    np.random.seed(42)

    w = np.random.randn(2, 1)
    x = 2 * np.random.rand(100, 1)
    y = 4 + 3 * x + np.random.randn(100, 1)

    plot_data(x, y)
    plot_regression(w, x, y, 0.01)
    plot_error(w, x, y, [0.001, 0.01, 0.1])
