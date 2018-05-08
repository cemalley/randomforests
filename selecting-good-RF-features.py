#!/usr/bin/env python3


# Code from http://blog.datadive.net/selecting-good-features-part-iii-random-forests/
# Changed a bit for python3


from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
import numpy as np
boston = load_boston()
X = boston["data"]
Y = boston["target"]
names = boston["feature_names"]
rf = RandomForestRegressor()
rf.fit(X, Y)
print("Features sorted by their score:")
print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), reverse=True))
size = 10000
size
np.random.seed(seed=10)
np.random.seed()
np.random.seed(seed=10)
X_seed = np.random.normal(0, 1, size)
X0 = X_seed + np.random.normal(0, .1, size)
X1 = X_seed + np.random.normal(0, .1, size)
X2 = X_seed + np.random.normal(0, .1, size)
X = np.array([X0, X1, X2]).T
X
Y = X0 + X1 + X2
rf = RandomForestRegressor(n_estimator=20, max_features=2)
rf = RandomForestRegressor(n_estimators=20, max_features=2)
rf.fit(X, Y)
print("Scores for X0, X1, X2:", list(map(lambda x:round (x,3),rf.feature_importances_)) )
