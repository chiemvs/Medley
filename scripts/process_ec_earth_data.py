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
from Medley.utils import chunk_func, spi_from_monthly_rainfall, fit_gamma, transform_to_spi, regions
from Medley.preprocessing import monthly_resample_func, makemask
from Medley.dataloading import datapath

def process_one(filepath, nyearshift: int = 0):
    """
    nyearshift for concatenation
    """
    # Drop first year
    da = xr.open_dataset(filepath)['pr'].isel(time = slice(12,None))
    # Fixing only positive coordinates
    da = da.assign_coords({'lon': [l - 360 if l >= 180 else l for l in da.lon.values]}).sortby('lon')
    # Left stamping and shifting the contiguous 
    first = pd.Timestamp(da.time.values[0])
    newaxis = pd.date_range(start = f'{first.year + nyearshift}-{first.strftime("%m")}-01', periods = len(da.time), freq = 'MS')
    da = da.assign_coords({'time':newaxis})
    da.close()
    return da, newaxis


ecpath = Path('/scistor/ivm/data_catalogue/climate_models/amip_style/ec_earth/')
amoc_strs = {'17sv': 99,'14sv': 87,'07sv':-30} # Shifts to harmonize the time axes to 1950-1959

"""
Making ec-earth landsea mask
"""
#land = xr.open_dataarray('/scistor/ivm/data_catalogue/reanalysis/ERA5_0.25/SST/sst_nhplus_june2sept.nc')[0,:,:].isnull()
#land.name = 'land'
#land = land.rename({'longitude':'lon','latitude':'lat'}).drop('time')
#ecearth, _ = process_one(ecpath / 'pr_Amon_EC-Earth3_07sv_r10i1p1f1_gr_197901-198912.nc')
#ecearth = ecearth.isel(time = 0).drop('time')
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
EC earth precipitations to 
concatenate multiple members
And also multiple experiments (so they are on the same SPI scale and we can see drying)
""" 
landseamask = xr.open_dataarray(ecpath / 'landseamask.nc')
mask = makemask(regions['medwest']).rename({'longitude':'lon','latitude':'lat'}).reindex_like(landseamask, method = 'nearest')
land_in_region = np.logical_and(mask,landseamask).stack({'latlon':['lat','lon']})


arrs = []
for amoc_str, shift in amoc_strs.items():
    files = np.sort(list(ecpath.glob(f'pr_Amon*_{amoc_str}_*')))
    subarrs = []
    for i, filepath in enumerate(files):
        da, axis = process_one(filepath, nyearshift = shift)
        #da, axis = process_one(filepath, nyearshift = 0)
        da = da.stack({'latlon':['lat','lon']}).sel(latlon = land_in_region)
        da = da.expand_dims({'amoc':[amoc_str],'member':[i]})
        subarrs.append(da)
    arrs.append(xr.concat(subarrs, 'member'))
        
#index = pd.MultiIndex.from_product([amoc_strs,pd.RangeIndex(20)], names = ['amoc','run'])
#index.name = 'member'
total = xr.concat(arrs, 'amoc') #.rename({'concat_dim':'member'})
 
spi = total.copy()
spi.encoding = {}
spi.attrs.update({'standard_name':'SPI-1','long_name':'standardized_precipitation_index','units':'std'})
for latlon in total.latlon:
    spi.loc[:,:,:,latlon] = transform_to_spi(total.loc[:,:,:,latlon])

outpath = datapath / 'ecearth'
spi.unstack('latlon').sortby('lat').sortby('lon').to_netcdf(outpath / 'spi1_medwest.nc')
spi.mean('latlon').to_netcdf(outpath / 'spi1_medwest_mean.nc')
# Apply ufunc does not work (only pure numpy)
#spis = xr.apply_ufunc(transform_to_spi, testarrs, 
#        #exclude_dims = set(('time',)), # Would be dropped from output (rank day anomalies) as well
#        input_core_dims=[['time']],
#        output_core_dims=[['time']], 
#        kwargs={'minsamples':20},
#        vectorize = True, dask = "parallelized",
#        output_dtypes=[np.float64],
#        dask_gufunc_kwargs = dict()) # rank, latitude, longitude



