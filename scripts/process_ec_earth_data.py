import sys
import os
import zarr
import xarray as xr
import numpy as np
import pandas as pd
import pyarrow as pa

from pathlib import Path
from typing import Union
from scipy import stats

sys.path.append(os.path.expanduser('~/Documents/Medley/'))
from Medley.utils import chunk_func, spi_from_monthly_rainfall, fit_gamma, transform_to_spi
from Medley.preprocessing import monthly_resample_func
from Medley.dataloading import datapath

"""
Making ec-earth landsea mask
"""
#land = xr.open_dataarray('/scistor/ivm/data_catalogue/reanalysis/ERA5_0.25/SST/sst_nhplus_june2sept.nc')[0,:,:].isnull()
#land.name = 'land'
#land = land.rename({'longitude':'lon','latitude':'lat'}).drop('time')
#land = land.assign_coords({'lon': [l + 360 if l < 0 else l for l in land.lon.values]}).sortby('lon') # Only positive longitudes, and monotonically increasing
#ecearth = xr.open_dataset('/scistor/ivm/data_catalogue/climate_models/amip_style/ec_earth/pr_Amon_EC-Earth3_07sv_r10i1p1f1_gr_197901-198912.nc')['pr'][0,:,:]
#landseamask = land.reindex_like(ecearth, method = 'nearest')
#landseamask.to_netcdf('/scistor/ivm/data_catalogue/climate_models/amip_style/ec_earth/landseamask.nc')

"""
ECAD testing code
"""
testdf = pd.read_hdf(datapath / f'eca_preaggregated_RR.h5').dropna(axis = 1, how = 'all').iloc[:,:100]
testdf.index.name = 'time'
testts = testdf[21] 

#spits = transform_to_spi(testts, climate_period = slice('1990-01-01','2000-01-01'))
spits = transform_to_spi(testts)
#test = testdf.apply(transform_to_spi, axis = 0)

"""
EC-earth testing code
"""
# Would I want to concatenate multiple members?, yes because only 11 years each (of which 10 are independent)
testarrs = xr.open_dataset('/scistor/ivm/data_catalogue/climate_models/amip_style/ec_earth/pr_Amon_EC-Earth3_07sv_r10i1p1f1_gr_197901-198912.nc')['pr'][:,:10,:10]
testarr = testarrs[:,1,1]

spimontharr = transform_to_spi(testarr, minsamples = 10)

"""
EC earth precipitations to 
concatenate multiple members
And also multiple experiments (so they are on the same SPI scale and we can see drying)
""" 
ecpath = Path('/scistor/ivm/data_catalogue/climate_models/amip_style/ec_earth/')
landseamask = xr.open_dataarray(ecpath / 'landseamask.nc')
files = np.sort(list(ecpath.glob('pr_Amon*')))

def process_one(filepath):
    
    amoc_str = filepath.name.split('_')[3]
    # Drop first year
    da = xr.open_dataset(filepath)['pr'].isel(time = slice(12,None))
    return da

    
# Apply ufunc does not work (only pure numpy)
#spis = xr.apply_ufunc(transform_to_spi, testarrs, 
#        #exclude_dims = set(('time',)), # Would be dropped from output (rank day anomalies) as well
#        input_core_dims=[['time']],
#        output_core_dims=[['time']], 
#        kwargs={'minsamples':20},
#        vectorize = True, dask = "parallelized",
#        output_dtypes=[np.float64],
#        dask_gufunc_kwargs = dict()) # rank, latitude, longitude



