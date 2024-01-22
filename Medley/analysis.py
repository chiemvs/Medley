import os
import sys
import numpy as np 
import pandas as pd 
import xarray as xr
from scipy.stats import linregress

def xr_linregress(y, x, min_count = 30):
    """
    Fit between two 1D arrays: y = a + b*x
    Returns 1D array [a, b]
    Removal of NaNs because scipy linregress cannot handle them
    Also setting a minimum amount of samples
    """
    isnan = np.isnan(y)
    y = y[~isnan]
    x = x[~isnan]
    if len(y) >= min_count:
        reg_result = linregress(x = x, y = y)
        result = [reg_result[1], reg_result[0]] # intercept, slope unit y / unit x]
    else:
        result = [np.nan, np.nan]
    return np.array(result, dtype = np.float32) 

def trendfit_robust(da : xr.DataArray, standardize: bool = True, min_count: int = 30):
    """
    Using ufunc on array of which one dimension should be 'time' or 'year',
    returns same array but with this dimension replaced by 
    dimension of length two: [intercept, slope] slope is per year
    """
    # Setting up an x variable.
    if not ('year' in da.dims):
        x_year = da.time.dt.year # retains the dimension name 'time'
        do_not_broadcast = 'time'
    else:
        x_year = da.year # Dimension name itself is also 'year'
        do_not_broadcast = 'year'
    coefs = xr.apply_ufunc(xr_linregress, da, x_year, exclude_dims = set((do_not_broadcast,)),
            input_core_dims=[[do_not_broadcast],[do_not_broadcast]],
            output_core_dims=[['what']], 
            vectorize = True, dask = "parallelized",
            output_dtypes=[np.float32],
            dask_gufunc_kwargs = dict(output_sizes={'what':2}),
            kwargs = dict(min_count = min_count))
    coefs.coords.update({'what':['intercept','slope']})
    if standardize:
        coefs = coefs / da.std(do_not_broadcast) # Cannot do skipna = False because then many slices would fail.
        coefs.attrs.update({'units':'std/yr'})
    else:
        coefs.attrs.update({'units':f'{da.attrs["units"]}/yr'})
    return coefs
