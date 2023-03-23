import os
import sys
import dask
import zarr
import array
import warnings
import numpy as np
import xarray as xr
import pandas as pd

from pathlib import Path

sys.path.append(os.path.expanduser('~/Documents/Medley'))
from scripts.prepare_monthly_ts_data import datapath

def zonal_u250_era5(lonmin: float = None, lonmax: float = None, latmin: float = None, latmax: float = None) -> xr.DataArray:
    """
    Extracting monthly average zonal mean u zonal wind 
    defaulting to entire domain, but possible within limited domain
    lonmin and lonmax are in degrees east
    latmin and latmax are in degrees north
    Using daily u250 data from Tamara  (latitude = decreasing, longitude = increasing, positive only)
    Outputting 2D array (time [months], latitude [degrees north])
    """
    da = xr.open_dataset('/scistor/ivm/data_catalogue/reanalysis/ERA5_0.25/u_winds_250/1959-2021_u_winds_250hPa.nc', chunks = {'latitude':4})['u']
    # longitudinal limitation
    if lonmin is None:
        lonslice = slice(lonmin,lonmax,None)
    elif lonmin < 0: # Correcting for the fact that we have positive only coordinates
        lonmin = 360 + lonmin # Slice will not be contiguous so selecting by index 
        lonslice = np.concatenate([da.longitude.values[da.longitude >= lonmin],da.longitude.values[da.longitude <= lonmax]])
    else:
        lonslice = slice(lonmin,lonmax,None)
    # latitudinal limitation
    latslice = slice(latmax, latmin, None)  # stored with Decreasing latitude
    da = da.sel({'longitude':lonslice, 'latitude':latslice})
    monthly = da.mean('longitude').resample(time = 'M', label = 'left').mean()
    monthly.coords['time'] = monthly.time + pd.Timedelta('1D') # Offsetting the label by 1 day
    monthly.attrs = da.attrs
    monthly.attrs.update({'resample':'monthly_mean'})
    return monthly

lonmin = -25
lonmax = 50
da = zonal_u250_era5(lonmin = lonmin, lonmax = lonmax, latmin = 0, latmax = None)
da.to_netcdf( datapath / f'monthly_zonalmean_u250_NH_{lonmin}E_{lonmax}E.nc')
