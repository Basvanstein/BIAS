from __future__ import annotations

from collections import OrderedDict
from typing import List, Union

import numpy as np
from joblib import Parallel, delayed
from numpy import array
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble._base import _partition_estimators
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

from ..solution import Solution

__authors__ = ["Hao Wang"]

class s0:
    "random surrogate function"

    def __init__(self):
        self.is_fitted = False

    def fit(self, X: Union[Solution, List, np.ndarray], y: np.ndarray):
        self.is_fitted = True
        return True

    def _check_X(self, X: Union[Solution, List, np.ndarray]) -> Solution:
        X_ = array(X, dtype=object)
        if hasattr(self, "_levels"):
            X_cat = self._enc.fit_transform(X_[:, self._cat_idx])
            X = np.c_[np.delete(X_, self._cat_idx, 1).astype(float), X_cat]
        return X

    def predict(self, X: Union[Solution, List, np.ndarray], eval_MSE=False) -> np.ndarray:
        y_hat = np.random.uniform(size=len(X))
        mse = np.ones(len(X)) * np.random.uniform(size=len(X))
        return (y_hat , mse) if eval_MSE else y_hat

