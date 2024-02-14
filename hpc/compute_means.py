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
from Medley.utils import udomains

"""
Data-wrangling for monthly jet indices based on daily zonal winds from ERA5
at two tropospheric pressure levels, which are pre-downloaded on the cluster. 
This is separate from the effort in scripts/retrieve_monthly_era5
which is concerned with the stratospheric vortex (which is not pre-downloaded and had to be retrieved).
"""

def zonal_u_era5(level = 250, lonmin: float = None, lonmax: float = None, latmin: float = None, latmax: float = None) -> tuple[xr.DataArray,xr.DataArray]:
    """
    Extracting monthly average zonal mean u zonal wind 
    defaulting to entire domain, but possible within limited domain
    lonmin and lonmax are in degrees east
    latmin and latmax are in degrees north
    Using daily u250 data from Tamara  (latitude = decreasing, longitude = increasing, positive only)
    Outputting one 2D array (time [months], latitude [degrees north])
    Outputting one 1D array (time [months])
    """
    assert (level in [250,500]), "Only 500hpa and 250 hpa are included"
    basepath = Path('/scistor/ivm/data_catalogue/reanalysis/ERA5_0.25')
    subpath = basepath / f'u_winds_{level}' 
    #da = xr.open_dataset(subpath / f'1959-2021_u_winds_{level}hPa.nc', chunks = {'latitude':4})['u'] 
    da = xr.open_dataset(subpath / f'1959-2021_u_winds_{level}hPa.nc')['u'] # No efficient chunking exists for the zonal mean of latmax
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
    monthly = da.resample(time = 'M', label = 'left').mean()
    monthly.coords['time'] = monthly.time + pd.Timedelta('1D') # Offsetting the label by 1 day
    zonal_u_mean = monthly.mean('longitude')
    zonal_u_mean.attrs = da.attrs
    zonal_u_mean.attrs.update({'resample':'monthly_mean'})
    zonal_lat_mean = monthly.idxmax('latitude').mean('longitude') # Order presented by Albert Osso
    zonal_lat_mean.attrs = da.attrs
    zonal_lat_mean.attrs.update({'resample':'monthly_mean'})
    return zonal_u_mean, zonal_lat_mean


if __name__ == '__main__':
    level = 250
    datapath = datapath / 'era5'
    for lonmin, lonmax in udomains.values():
        zon_u, zon_lat = zonal_u_era5(level = level, lonmin = lonmin, lonmax = lonmax, latmin = 0, latmax = None)
        zon_u.to_netcdf( datapath / f'monthly_zonalmean_u{level}_NH_{lonmin}E_{lonmax}E.nc')
        zon_lat.to_netcdf( datapath / f'monthly_zonallatmax_u{level}_NH_{lonmin}E_{lonmax}E.nc')
    
