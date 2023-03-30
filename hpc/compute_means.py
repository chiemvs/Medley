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
from scripts.prepare_monthly_ts_data import datapath, udomains

def zonal_u_era5(level = 250, lonmin: float = None, lonmax: float = None, latmin: float = None, latmax: float = None) -> xr.DataArray:
    """
    Extracting monthly average zonal mean u zonal wind 
    defaulting to entire domain, but possible within limited domain
    lonmin and lonmax are in degrees east
    latmin and latmax are in degrees north
    Using daily u250 data from Tamara  (latitude = decreasing, longitude = increasing, positive only)
    Outputting 2D array (time [months], latitude [degrees north])
    """
    assert (level in [250,500]), "Only 500hpa and 250 hpa are included"
    basepath = Path('/scistor/ivm/data_catalogue/reanalysis/ERA5_0.25')
    if level == 250: # Different directory structure
        subpath = basepath / 'u_winds_250' 
    else:
        subpath = basepath / 'u_winds'
    da = xr.open_dataset(subpath / f'1959-2021_u_winds_{level}hPa.nc', chunks = {'latitude':4})['u']
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


if __name__ == '__main__':
    level = 500
    for lonmin, lonmax in udomains.values():
        da = zonal_u_era5(level = level, lonmin = lonmin, lonmax = lonmax, latmin = 0, latmax = None)
        da.to_netcdf( datapath / f'monthly_zonalmean_u{level}_NH_{lonmin}E_{lonmax}E.nc')
    
