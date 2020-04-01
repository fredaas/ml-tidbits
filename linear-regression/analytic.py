#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams

rcParams["font.family"] = "monospace"

def mse(w, x, y_pred, y):
    m = len(y)
    return (1 / (2 * m)) * np.sum(np.square(y_pred - y))

def weights(x, y):
    m = np.size(x)

    mean_x, mean_y = np.mean(x), np.mean(y)
    SS_xy = np.sum(y * x) - m * mean_y * mean_x
    SS_xx = np.sum(x * x) - m * mean_x * mean_x

    w_1 = SS_xy / SS_xx
    w_0 = mean_y - w_1 * mean_x

    return w_0, w_1

def plot_regression(x, y):
    w = weights(x, y)
    y_pred = w[0] + w[1] * x
    error = mse(w, x, y_pred, y)

    print("Error: {:.4}".format(error))
    print("Weights: {:.4} {:.4}".format(w[0], w[1]))

    _, ax = plt.subplots(figsize=(5, 5))

    ax.plot(x, y_pred, "r-", linewidth=.25, antialiased=True)
    ax.plot(x, y, "x")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend([ "h(x)" ])
    ax.set_ylim([0, 12])

    plt.title("Linear Regression (Analytic)")
    plt.grid(linestyle="dotted")
    plt.tight_layout()

    plt.savefig("analytic.pdf", format="pdf")

if __name__ == "__main__":
    np.random.seed(42)

    x = 2 * np.random.rand(100, 1)
    y = 4 + 3 * x + np.random.randn(100, 1)

    plot_regression(x, y)
