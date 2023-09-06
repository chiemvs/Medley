import os
import sys
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression

class LinearTrendRegressor(LinearRegression):
    def __init__(self):
        super().__init__(fit_intercept = True)

    def _extract_year_from_X(self, X):
        """
        Uses time information in the index of X to create the
        explanatory variable for the trend
        """
        if isinstance(X.index, pd.DatetimeIndex):
            Xyear = X.index.year
        else: # assuming integer based index with anchor_year
            Xyear = X.index.get_level_values('anchor_year')
        Xyear = Xyear.values[:,np.newaxis] # 2D as X-format for sklearn models
        return Xyear 

    def fit(self, X, y, sample_weight = None):
        Xyear = self._extract_year_from_X(X)
        super().fit(X = Xyear, y = y)
        return self

    def predict(self, X):
        Xyear = self._extract_year_from_X(X)
        yhat = super().predict(X = Xyear)
        return yhat

class RandomForestResidualRegressor:
    def __init__(self):
        pass

estimators = {'rfreg':RandomForestRegressor,
        'rfresreg':RandomForestResidualRegressor,
        'rfclas':RandomForestClassifier,
        'linreg':LinearRegression,
        'climreg':LinearTrendRegressor}

def return_estimator(estimator:str):
    assert estimator in estimators.keys(), f'choose one of {list(estimators.keys())}'
    return estimators[estimator]

