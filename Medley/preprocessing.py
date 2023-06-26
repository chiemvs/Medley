import zarr
import numpy as np
import pandas as pd
import xarray as xr

class Anomalizer(object):

    def __init__(self):
        pass

    def fit(self, X : pd.DataFrame):
        """
        Fitting on X, do not provide holdout  
        """
        self.climate = X.groupby(X.index.month).mean()

    def transform(self, X: pd.DataFrame):
        new = X.copy()
        new.index = new.index.month
        new -= self.climate.loc[new.index,X.columns]
        new.index = X.index
        return new 

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
        

def monthly_resample_func(ds: xr.Dataset, how = 'mean') -> xr.Dataset:
    """
    Left stamping (first day of the month)
    """
    resampled_object = ds.resample(time = 'M', label = 'left')
    f = getattr(resampled_object, how)
    newds = f(skipna = False)
    newds.coords['time'] = newds.time + pd.Timedelta('1D') # Shifting due to weird label behaviour.
    for varname in (varname for varname in newds.variables if ('time' in newds[varname].dims)):
        newds[varname].attrs.update({'resample':f'monthly_{how}'})
    return newds
