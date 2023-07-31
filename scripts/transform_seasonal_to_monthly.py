import os
import sys
import zarr
import array
import fsspec
import warnings
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import xarray as xr
import pandas as pd

from pathlib import Path
from datetime import datetime, timedelta

sys.path.append(os.path.expanduser('~/Documents/Medley'))
from scripts.prepare_monthly_ts_data import udomains

seas_datapath = Path('/scistor/ivm/jsn295/Medi/seasonal')
mon_datapath = Path('/scistor/ivm/jsn295/Medi/monthly')

"""
Alter Eddy Kinetic Energy data (vertically integrated by mass) from Rei Chemke
Notes:
    - units are missing
    - ERA-I has artificial zeros in SON 2019
From seasonal values to monthly values (smearing out the average)
6 Reanalyses
"""

def zonal_eke(lonmin: float = None, lonmax: float = None, latmin: float = None, latmax: float = None) -> xr.DataArray:
    """
    Extracting seasonal average zonal mean EKE 
    defaulting to entire domain, but possible within limited domain
    lonmin and lonmax are in degrees east
    (latitude = increasing, longitude = increasing, positive only)
    Outputting 2D array (time [months], latitude [degrees north])
    timestamps are given as firstday of the month
    """
    filepath = seas_datapath/ 'vrt_EKE_butw_25_6.nc'
    ds = xr.open_dataset(filepath)
    reanalysis_coords = ''.join(ds.Reanalyses.astype(str).values).split(', ') # Weird way of saving, as one array of letters.
    ds.coords.update({'Reanalysis':reanalysis_coords,'longitude':ds.lon,'latitude':ds.lat})
    ds = ds.drop('Reanalyses')
    # longitudinal limitation
    if lonmin is None:
        lonslice = slice(lonmin,lonmax,None)
    elif lonmin < 0: # Correcting for the fact that we have positive only coordinates
        lonmin = 360 + lonmin # Slice will not be contiguous so selecting by index 
        lonslice = np.concatenate([ds.longitude.values[ds.longitude >= lonmin],ds.longitude.values[ds.longitude <= lonmax]])
    else:
        lonslice = slice(lonmin,lonmax,None)
    # latitudinal limitation
    latslice = slice(latmin, latmax, None)  # stored with Decreasing latitude

    averages = []
    for seas, startmonth in {'djf':'JAN','mam':'MAR','jja':'JUN','son':'SEP'}.items():
        da = ds[f'eke_trans_vrt_{seas}_rean_buttercup_25_6']
        da = da.sel({'longitude':lonslice, 'latitude':latslice})
        zonmean = da.mean('longitude')
        for i in range(len(seas)):  
            temp = zonmean.copy()
            timestamps = pd.date_range('1979-01-01', periods = len(da.year), freq = f'AS-{startmonth}')
            if seas == 'djf':
                timestamps += pd.tseries.offsets.MonthBegin(i - 1) # DJF stamped with year of the JAN and FEB.
            else:
                timestamps += pd.tseries.offsets.MonthBegin(i)
            temp.coords.update({'year':timestamps})
            averages.append(temp)
    averages = xr.concat(averages, dim = 'year').rename({'year':'time'}).sortby('time')
        
    # ERA interim has artificial zero's in SON 2019 that need to be set to nan
    averages.loc['ERA-I',slice('2019-09-01','2019-11-01')] = np.nan
    averages.name = 'eke'
    return averages


if __name__ == '__main__':
    for lonmin, lonmax in udomains.values():
        da = zonal_eke(lonmin = lonmin, lonmax = lonmax, latmin = 0, latmax = None)
        da.to_netcdf( mon_datapath / f'monthly_zonalmean_EKE_NH_{lonmin}E_{lonmax}E.nc')
    
