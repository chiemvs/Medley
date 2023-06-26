import warnings
import array
import numpy as np
import xarray as xr
import pandas as pd

from datetime import datetime, timedelta

def data_for_pcolormesh(array, shading:str):
    """Xarray array to usuable things"""
    lats = array.latitude.values # Interpreted as northwest corners (90 is in there)
    lons = array.longitude.values # Interpreted as northwest corners (-180 is in there, 180 not)
    if shading == 'flat':
        lats = np.concatenate([lats[[0]] - np.diff(lats)[0], lats], axis = 0) # Adding the sourthern edge 
        lons = np.concatenate([lons, lons[[-1]] + np.diff(lons)[0]], axis = 0)# Adding the eastern edge (only for flat shating)
    return lons, lats, array.values.squeeze()

def chunk_func(ds: xr.Dataset, chunks = {'latitude':50, 'longitude':50}) -> xr.Dataset:
    """
    Chunking only the spatial dimensions. Eases reading complete timeseries at one grid location. 
    """
    #ds = ds.transpose("time","latitude", "longitude")
    chunks.update({'time':len(ds.time)}) # Needs to be specified as well, otherwise chunk of size 1.
    return ds.chunk(chunks)

def decimal_year_to_datetime(decyear: float) -> datetime:
    """Decimal year to datetime, not accounting for leap years"""
    baseyear = int(decyear)
    ndays = 365 * (decyear - baseyear)
    return datetime(baseyear,1,1) + timedelta(days = ndays)

def process_ascii(timestamps: array.array, values: array.array, miss_val = -999.9) -> pd.DataFrame:
    """
    Missing value handling, and handling the case with multiple monthly values for one yearly timestamp
    """
    # Missing values
    values = np.array(values)
    values[np.isclose(values, np.full_like(values, miss_val))] = np.nan
    # Temporal index
    assert (np.allclose(np.diff(timestamps),1.0) or np.allclose(np.diff(timestamps), 1/12, atol = 0.001)), 'timestamps do not seem to be decimal years, with a yearly or monthly interval, check continuity'
    if len(timestamps) != len(values):
        assert (len(values) % len(timestamps)) == 0, 'values are not devisible by timestamps, check shapes and lengths'
        warnings.warn(f'Spotted one timestamp per {len(values)/len(timestamps)} values data')
    timestamps = pd.date_range(start = decimal_year_to_datetime(timestamps[0]), periods = len(values),freq = 'MS') # Left stamped
    series = pd.DataFrame(values[:,np.newaxis], index = timestamps)
    # Adding extra information, except for climexp file
    return series 
