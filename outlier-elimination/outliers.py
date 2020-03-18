#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams
from scipy import stats
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

rcParams['font.family'] = 'monospace'

outliers_frac = 0.25
num_samples   = 200
offsets       = [ 0, 2, 4 ]

num_inliers   = int((1. - outliers_frac) * num_samples)
num_outliers  = int(outliers_frac * num_samples)
ground_truth  = np.ones(num_samples, dtype=int)

ground_truth[-num_outliers:] = -1

xx, yy = np.meshgrid(np.linspace(-7, 7, 500), np.linspace(-7, 7, 500))

def oneclass_svm():
    return svm.OneClassSVM(
        nu=0.95 * outliers_frac + 0.05,
        kernel="rbf",
        gamma=0.1
    )

def robust_covariance():
    return EllipticEnvelope(contamination=outliers_frac)

def isolation_forest():
    return IsolationForest(
        max_samples=num_samples,
        contamination=outliers_frac,
        random_state=np.random.RandomState(42)
    )

classifiers = {
    "One-Class SVM": oneclass_svm(),
    "Robust Covariance": robust_covariance(),
    "Isolation Forest": isolation_forest()
}

_, axes = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(12, 12))

for j, offset in enumerate(offsets):
    np.random.seed(42)

    # Generate data
    X1 = 0.3 * np.random.randn(num_inliers // 2, 2) - offset
    X2 = 0.3 * np.random.randn(num_inliers // 2, 2) + offset
    X = np.r_[X1, X2]
    # Add outliers
    X = np.r_[X, np.random.uniform(low=-6, high=6, size=(num_outliers, 2))]

    for i, (clf_name, clf) in enumerate(classifiers.items()):
        # Fit data
        clf.fit(X)
        threshold = stats.scoreatpercentile(
            clf.decision_function(X), 100 * outliers_frac)
        y_pred = clf.predict(X)
        n_errors = (y_pred != ground_truth).sum()
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot
        ax = axes[j, i]
        ax.contourf(xx, yy, Z,
            levels=np.linspace(Z.min(), threshold, 7), cmap=plt.cm.Blues_r)
        ax.contour(xx, yy, Z,
            levels=[threshold], colors='red', linestyles="solid")
        ax.contourf(xx, yy, Z,
            levels=[threshold, Z.max()], colors='orange')
        b = ax.scatter(X[:-num_outliers, 0], X[:-num_outliers, 1],
            c='white', edgecolor='black')
        c = ax.scatter(X[-num_outliers:, 0], X[-num_outliers:, 1],
            c='black')
        ax.axis('tight')
        ax.legend(
            [b, c],
            ['Inliers', 'Outliers'],
            loc='lower right'
        )
        ax.set_title("%d. %s (errors: %d)" % (i + 1, clf_name, n_errors))
        ax.set_xlim((-7, 7))
        ax.set_ylim((-7, 7))

plt.tight_layout()
plt.savefig("outliers.png", format="png")
