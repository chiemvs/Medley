import sys
import os
import zarr
import fsspec
import xarray as xr
import numpy as np
import pandas as pd
import pyarrow as pa

from pathlib import Path

sys.path.append(os.path.expanduser('~/Documents/Medley/'))
from Medley.utils import chunk_func
from Medley.preprocessing import monthly_resample_func

"""
Script to put everying into zarr stores
"""



"""
WP3 data
SPEI-like Water surplus deficit 
"""
overwrite = False

ds = xr.open_zarr(fsspec.get_mapper('https://s3.bgc-jena.mpg.de:9000/xaida/SPEICube.zarr'), consolidated=True)
## Adding some information, and re-chunking
#for running_window in [30,90,180]:
#    ds[f'spei_{running_window}'].encoding.pop('chunks')
#    ds[f'spei_{running_window}'].encoding.pop('preferred_chunks')
#    ds[f'spei_{running_window}'].attrs.update({'long_name':'antecedent_water_surplus_or_deficit',
#            'units':'mm',
#            'calculation':'SUM(TP-PET)',
#            'window_length_days':running_window,
#            'potential_evap':'Penman_Monteith',
#            'source':'ERA5_and_Max_Planck_MPG',
#            'project':'XAIDA',
#            'workpackage':3,})
#
#localpath = Path('/scistor/ivm/data_catalogue/reanalysis/ERA5_0.25/WSD/1979_2021_water_surplus_deficit_daily.zarr')
## Writing out daily data for everyone.
#if (not localpath.exists()) or overwrite:
#    ds = chunk_func(ds)
#    ds.to_zarr(localpath)
#
## creating a monthly mediterranean subset.
## It is already aggregated/accumulated, but still sampling to monthly mean (mean of preceeding drought within a month, potentially extending until before that month)
#ds = xr.open_zarr(localpath)
ds = ds.sel(latitude = slice(65,20),longitude = slice(-20,55))
ds = monthly_resample_func(ds, how = 'mean')
ds = chunk_func(ds)
outpath = Path('/scistor/ivm/jsn295/Medi/monthly/1979_2021_monthly_water_surplus_deficit.zarr')
if (not outpath.exists()) or overwrite:
    ds.to_zarr(outpath)

"""
WP3 data
Label and event cubes for spei/wsd, are daily
Downloaded manually from https://nextcloud.bgc-jena.mpg.de/s/MTc2BNJNkoSMxdH. Cannot be resampled to months because consists of integers. Just combining them into a single store, like the SPEI-cube
"""

downloadpath = Path('/scistor/ivm/jsn295/Medi/daily/cloud_download/')

naming = {'EventCube':lambda window: f'EventCube_Univariate_spei_{window}_0.05.zarr',
        'LabelCube':lambda window: f'LabelCube_Univariate_spei_{window}_0.05_Europe_1979_2021.zarr'}
for name in naming.keys():
    key_value = {}
    for window in [30,90,180]:
        da = xr.open_zarr(downloadpath / naming[name](window), consolidated = False)['layer']
        da = da.sel(latitude = slice(65,20),longitude = slice(-20,55))
        da.encoding.pop('chunks')
        da.encoding.pop('preferred_chunks')
        da.attrs.update({'window_length_days':window,'threshold':0.05})
        key_value.update({f'spei_{window}':da})
    ds = xr.Dataset(key_value)
    ds = chunk_func(ds)
    outpath = downloadpath.parent / f'1979_2021_water_surplus_deficit_daily_{name}.zarr'
    if (not outpath.exists()) or overwrite:
        ds.to_zarr(outpath, consolidated=True)

"""
WP3 data is not seamasked
so create one from ERA5 ssts (also resolution 0.25)
"""
example = xr.open_zarr('/scistor/ivm/jsn295/Medi/monthly/1979_2021_monthly_water_surplus_deficit.zarr').isel(time = 0)
whereland = xr.open_dataarray('/scistor/ivm/jsn295/ERA5/sst_nhplus.nc')[0,:,:].isnull()
whereland.name = 'land'
whereland = whereland.reindex_like(example, method = 'nearest').drop('time')
whereland.to_netcdf('/scistor/ivm/jsn295/Medi/landseamask_wp3.nc',encoding={'land':{'dtype':'byte'}})
