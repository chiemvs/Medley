import zarr
import lilio
import numpy as np
import pandas as pd
import xarray as xr

from typing import Union
from pathlib import Path

from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


"""
Resampling functionalities to set up a statistical prediction setup (X, y)
based on e.g. monthly y in Jan, Feb, March, resulting in multiple samples per year (strategy = multi),
or mean of monthly y over months Jan, Feb, March, resulting in one sample per year (strategy = single),
both in combination with lagged drivers X. Builds on the lilio package for calendar-based resampling
also functionality for standardization or removing seasonality from timeseries 
"""

class Anomalizer(BaseEstimator):
    def __init__(self, startyear : int = None, endyear: int = None):
        """
        Anomalizer by subtracting month-in-year climatological value
        Not really needed if using SPI, perhaps only for predictors
        Possible to set the baseperiod (endyear = inclusive),
        if not set (default) then all available data is used to estimate climate
        Should be able to anomalize both  
        """
        self.startyear = startyear
        self.endyear = endyear

    def __repr__(self):
        return f'Anomalizer(startyear = {self.startyear}, endyear = {self.endyear})'

    def _get_subsetter(self):
        """
        Logic called to build a slice object for an X based on
        years set at initialization
        returns None if these are not set
        """
        isinput = [ (not y is None) for y in [self.startyear, self.endyear]]
        if all(isinput): 
            startyear = f'{self.startyear}-01-01'
            endyear = f'{self.endyear}-12-31'
            subset = slice(pd.Timestamp(startyear),pd.Timestamp(endyear))
        elif sum(isinput) == 1:
            raise NotImplementedError('set both start and endyear of the Anomalizer, or none at all')
        else:
            subset = None
        return subset

    def _get_month_index(self, X : pd.DataFrame):
        if isinstance(X.index, pd.DatetimeIndex):
            indexer = X.index.month
        elif 'time' in X.index.names: # MultiIndex
            indexer = X.index.get_level_values('time').month
        elif 'anchor_month' in X.index.names: # Multi resampling
            indexer = X.index.get_level_values('anchor_month')
        else:
            raise NotImplementedError('Month-based anomalizing not compatible with single target resampling with only one value per year. Consider anomalizing before resampling')
        return indexer

    def fit(self, X : pd.DataFrame, y = None):
        """
        Fitting on X, do not provide holdout  
        """
        subset = self._get_subsetter()
        if not (subset is None):
            if isinstance(X.index, pd.DatetimeIndex):
                X = X.loc[subset,:] 
            else: # Indexed by anchor_year
                X = X.loc[slice(subset.start.year,subset.stop.year + 1),:]
        grouper = self._get_month_index(X)
        self.climate = X.groupby(grouper, axis = 0).mean()
        return self

    def transform(self, X: pd.DataFrame):
        new = X.copy()
        new.index = self._get_month_index(X) # creates double non-unique values in index
        new -= self.climate.loc[new.index,X.columns]
        new.index = X.index # reset original index
        return new 

    def fit_transform(self, X, y = None):
        self.fit(X)
        return self.transform(X)

def make_pipeline(estimator, anom = False, anom_kwargs = dict(), scale = False, scale_kwargs = dict()) -> Pipeline:
    """
    Initializes preprocessing objects and puts those in a pipeline
    with an initialized estimator as final step
    """
    assert not ((not anom) and (not scale)), 'No need to construct a pipeline if anom and std are False'
    pipeline = list()
    if anom:
        pipeline.append(('anom',Anomalizer(**anom_kwargs)))
    if scale:
        pipeline.append(('scale',StandardScaler(**scale_kwargs)))
    pipeline.append(('estimator',estimator))
    return Pipeline(pipeline)

def monthly_resample_func(ds: xr.Dataset, how = 'mean') -> xr.Dataset:
    """
    Used in data retrieval and wrangling, not for setting up statistical prediction
    Left stamping (first day of the month)
    """
    resampled_object = ds.resample(time = 'M', label = 'left')
    f = getattr(resampled_object, how)
    newds = f(skipna = False)
    newds.coords['time'] = newds.time + pd.Timedelta('1D') # Shifting due to weird label behaviour.
    for varname in (varname for varname in newds.variables if ('time' in newds[varname].dims)):
        newds[varname].attrs.update({'resample':f'monthly_{how}'})
    return newds

def simultaneous_resample(X : pd.DataFrame, y : pd.DataFrame, firstmonth = 12, lastmonth = 1, average = True):
    """
    Resampling not for lagged statistical prediction (e.g. y_t=0 and X_t=-1) 
    but to quantify concurrent/simultaneous assocation in a range of months 
    defined by firstmonth and lastmonth.
    lastmonth is inclusive
    multiple monthly samples if average = False
    """
    combined = X.join(y, how = 'outer')
    if firstmonth > lastmonth:
        months = np.arange(firstmonth - 12, lastmonth + 1, 1)
        months  = np.where(months <=0, months + 12, months)
    else:
        months = np.arange(firstmonth, lastmonth + 1, 1)
    if average:
        cal = lilio.Calendar(anchor = f'{firstmonth}-01')
        cal.add_intervals("precursor", length=f"{len(months)}M", n=1, gap = f'-{len(months)}M') # Fully co-occurring
    else:
        cal = lilio.Calendar(anchor = f'{lastmonth}-01')
        cal.add_intervals("precursor", length="1M", n=1, gap = '-1M') # Fully co-occurring
        if len(months) > 1:
            cal.add_intervals("precursor", length="1M", n=len(months)-1, gap = '0M') # Fully co-occurring
    cal.map_years(start=combined.index.year.min() - 2, end=combined.index.year.max() + 2)
    res = lilio.resample(cal, combined, how = 'nanmean')
    res = res.set_index(['anchor_year','i_interval','interval']).drop('is_target', axis = 1) # Lilio bookkeeping
    X_res = res[X.columns]
    y_res = res[y.columns]

    return X_res, y_res, cal

def single_target_lagged_resample(X: pd.DataFrame, y: pd.DataFrame, separation = 0, firstmonth = 12, lastmonth = 12, n = 1, precursor_agg = 1, allow_precursor_overlap = True):
    """
    resampling with one target period per calendar year
    Separation between target and first precursor is in terms of number of months.
    Always an averaging is applied.
    Averaging period for target is determined by firstmonth / lastmonth interval (lastmonth is inclusive)
    Averaging period for precursor is given as integer in number of months
    if precursor_agg > 1, you can opt for allowing overlap. 
    """
    combined = X.join(y, how = 'outer')
    if firstmonth > lastmonth:
        months = np.arange(firstmonth - 12, lastmonth + 1, 1)
        months  = np.where(months <=0, months + 12, months)
    else:
        months = np.arange(firstmonth, lastmonth + 1, 1)
    cal = lilio.Calendar(anchor = f'{firstmonth}-01')
    cal.add_intervals("target", length=f"{len(months)}M")
    counter = n
    while counter > 0:
        if counter == n: # First precursor
            gap = separation
        elif (precursor_agg > 1) and allow_precursor_overlap:
            gap = -(precursor_agg - 1)
        else:
            gap = 0
        cal.add_intervals("precursor", length=f"{precursor_agg}M", n=1, gap = f'{gap}M')
        counter -= 1
    cal.map_years(start=combined.index.year.min() - 2, end=combined.index.year.max() + 2)
    res = lilio.resample(cal, combined, how = 'nanmean')
    res = res.set_index(['anchor_year','i_interval']).drop('interval', axis = 1)
    y_res = res.loc[res['is_target'],y.columns].unstack('i_interval')
    X_res = res.loc[~res['is_target'],:].drop(y.columns, axis = 1).drop('is_target', axis = 1).unstack('i_interval')
    X_res.columns.names = ['variables','i_interval']
    return X_res, y_res, cal # Also return the calendar for bookkeeping.

def multi_target_lagged_resample(X: pd.DataFrame, y: pd.DataFrame, separation = 0, firstmonth = 12, lastmonth = 3, target_agg = 1, n = 1, precursor_agg = 1):
    """
    Repeated calls of the single target lagged resampling, to create more samples 
    firstmonth and lastmonth take a different meaning (
    targets will be overlapping if target_agg > 1 
    precursors will certainly be overlapping.
    """
    if firstmonth > lastmonth:
        anchor_months = np.arange(firstmonth - 12, lastmonth + 1, 1)
        anchor_months  = np.where(anchor_months <=0, anchor_months + 12, anchor_months)
    else:
        anchor_months = np.arange(firstmonth, lastmonth + 1, 1)

    assert not (target_agg > len(anchor_months)), 'requested target aggregation too large for month range'
    firstmonths = anchor_months[:(len(anchor_months) - target_agg + 1)]
    lastmonths = anchor_months[(target_agg - 1):]
    
    Xlist = []
    ylist = []
    callist = []
    for start, end in zip(firstmonths,lastmonths):
        X1, y1, c1 = single_target_lagged_resample(X = X, y = y, firstmonth = start, lastmonth = end, n = n, precursor_agg = precursor_agg, separation = separation) 
        X1.index = pd.MultiIndex.from_product([X1.index, [start]], names = X1.index.names + ['anchor_month'])
        y1.index = X1.index # should match
        Xlist.append(X1)
        ylist.append(y1)
        callist.append(c1)

    return pd.concat(Xlist, axis = 0).sort_index(), pd.concat(ylist, axis = 0).sort_index(), callist


def makemask(region_dict) -> xr.DataArray:
    """
    Highest resolution (0.1 degree) gridded mask
    region dict is a nested dict containing (lonmin, latmin, lonmax, latmax) 
    example: 
    dict(include = {'italy':(8,35,18,45.63)},                                                                   exclude = {'eastadriatic':(15,43,20,48)})
    """
    rrmon = xr.open_zarr('/scistor/ivm/jsn295/Medi/monthly/rr_mon_ens_mean_0.1deg_reg_v27.0e.zarr/')['rr']
    example = rrmon.isel(time = 0, drop =True)
    mask = xr.DataArray(np.full_like(example, 0), coords = example.coords)
    def set_subset_to(lonmin, latmin, lonmax, latmax, array, value):
        lons = array.sel(longitude = slice(lonmin, lonmax)).longitude
        lats = array.sel(latitude = slice(latmin, latmax)).latitude
        array.loc[lats,lons] = value # cannot do .sel based assignment
    for args in region_dict['include'].values():
        set_subset_to(*args, array = mask, value = 1)
    for args in region_dict['exclude'].values():
        set_subset_to(*args, array = mask, value = 0)
    return mask

def mask_reduce(mask: xr.DataArray, data: Union[pd.DataFrame,xr.DataArray], 
        datalocs: pd.DataFrame = None, what: str = 'mean', return_masked: bool = False) -> pd.Series:
    """
    Gridded mask, data either station data or also gridded
    if station data then a separate frame should be supplied with locations
    indexed by station ids.
    """
    if isinstance(data, xr.DataArray):
        mask = mask.reindex_like(data, method = 'nearest') # Reindexing if possible resolution mismatch
        masked = xr.where(mask,data,np.nan)
        func = getattr(masked, what)
        result = func(['longitude','latitude']).to_pandas()
    else:
        assert not (datalocs is None), 'provide frame with station locations'
        ind = pd.MultiIndex.from_frame(datalocs[['LAT','LON']], names = ['latitude','longitude'])
        #ids = pd.Series(datalocs.index, index = ind)
        # Stacking and xr.DataArray.reindex fails (no nearest for multiindex), therefore loop
        within = pd.Series([int(mask.sel(latitude = lat, longitude = lon, method = 'nearest')) for lat,lon in ind], index = datalocs.index)
        within = within.loc[within == 1]
        masked = data.loc[:,within.index]
        func = getattr(masked, what)
        result = func(axis = 1)
    if return_masked:
        return result, masked
    else:
        return result

def average_within_mask(*args, minsamples: int = 1, **kwargs) -> pd.Series: 
    """
    averaging either stationdata or gridded data,
    based on gridded mask
    """
    series = mask_reduce(*args, **kwargs, what = 'mean')
    if minsamples > 1:
        counts = mask_reduce(*args, **kwargs, what = 'count')
        series.iloc[counts.values < minsamples] = np.nan
    return series

def remove_bottleneck(X: pd.DataFrame, y: pd.DataFrame, startyear: int = None, endyear: int = None, fraction_valid: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Several X variables have little data and would be bottlenecks when calling dropna(how = any)
    bottlenecks are:
    EKE only 1980-2018
    MJO only 1980-now
    AMOC only 2004-2020
    Therefore this function is aimed to first remove those
    and keep only those variables with a minimum fraction of valid data
    within a specified start/end window
    and then do dropna (both y and x-based)
    """
    # Years from which data is required, dropping bottleneck variables
    if isinstance(X.index, pd.DatetimeIndex):
        indexslice = X.index.year.slice_indexer(startyear,endyear)
        X = X.iloc[indexslice,:]
    else: # Assuming first level is an integer based year index (anchor year)
        yearslice = slice(startyear,endyear)
        X = X.loc[yearslice,:]
    insufficient = X.columns[X.count(axis = 0) < (len(X)*fraction_valid)]
    print(f'dropping predictors: {insufficient}')
    X = X.drop(insufficient, axis = 1).dropna()
    y = y.dropna()
    intersection = X.index.intersection(y.index)
    X = X.loc[intersection,:]
    y = y.loc[intersection,:]
    print(f'samples left: {y.size}')
    print(f'features left: {X.shape[1]}')
    return X, y
