import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge


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

class RandomForestResidualRegressor(RandomForestRegressor):
    def __init__(self, n_estimators = 100, max_depth = None, min_samples_split = 2, max_features = 1.0, **kwargs):
        """
        Passing of arguments is a bit inconvenient see:
        https://github.com/scikit-learn/scikit-learn/issues/13555 
        """
        super().__init__(n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split, max_features = max_features, **kwargs)

        self.trendregressor = LinearTrendRegressor()

    def fit(self, X, y, sample_weight = None):
        self.trendregressor.fit(X, y, sample_weight = sample_weight)
        yhat = self.trendregressor.predict(X)
        residuals = y - yhat
        super().fit(X = X, y = residuals)
        return self

    def predict(self, X):
        yhat = self.trendregressor.predict(X)
        residualshat = super().predict(X = X) 
        yhat += residualshat
        return yhat


_estimators = {'rfreg':RandomForestRegressor,
        'rfresreg':RandomForestResidualRegressor,
        'rfclas':RandomForestClassifier,
        'linreg':LinearRegression,
        'ridreg':Ridge,
        'climreg':LinearTrendRegressor,
        'xgbreg':xgb.XGBRegressor}

def return_estimator(estimator:str):
    assert estimator in _estimators.keys(), f'choose one of {list(estimators.keys())}'
    return _estimators[estimator]

