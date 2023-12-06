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
from xeofs.models import EOF

sys.path.append(os.path.expanduser('~/Documents/Medley/'))
from Medley.utils import chunk_func, spi_from_monthly_rainfall, fit_gamma, transform_to_spi, regions
from Medley.preprocessing import monthly_resample_func, makemask
from Medley.dataloading import datapath

def process_one(filepath, ncvarname: str, nyearshift: int = 0):
    """
    nyearshift for concatenation
    """
    # Drop first year
    da = xr.open_dataset(filepath)[ncvarname].isel(time = slice(12,None))
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
EC earth precipitations to 
concatenate multiple members
And also multiple experiments (so they are on the same SPI scale and we can see drying)
""" 
#landseamask = xr.open_dataarray(ecpath / 'landseamask.nc')
#mask = makemask(regions['medwest']).rename({'longitude':'lon','latitude':'lat'}).reindex_like(landseamask, method = 'nearest')
#land_in_region = np.logical_and(mask,landseamask).stack({'latlon':['lat','lon']})
#
#
#arrs = []
#for amoc_str, shift in amoc_strs.items():
#    files = np.sort(list(ecpath.glob(f'pr_Amon*_{amoc_str}_*')))
#    subarrs = []
#    for i, filepath in enumerate(files):
#        da, axis = process_one(filepath, ncvarname = 'pr', nyearshift = shift)
#        da = da.stack({'latlon':['lat','lon']}).sel(latlon = land_in_region)
#        da = da.expand_dims({'amoc':[amoc_str],'member':[i]})
#        subarrs.append(da)
#    arrs.append(xr.concat(subarrs, 'member'))
#        
#total = xr.concat(arrs, 'amoc')
# 
#spi = total.copy()
#spi.encoding = {}
#spi.attrs.update({'standard_name':'SPI-1','long_name':'standardized_precipitation_index','units':'std'})
#for latlon in total.latlon:
#    spi.loc[:,:,:,latlon] = transform_to_spi(total.loc[:,:,:,latlon])
#
#outpath = datapath / 'ecearth'
#spi.unstack('latlon').sortby('lat').sortby('lon').to_netcdf(outpath / 'spi1_medwest.nc')
#spi.mean('latlon').to_netcdf(outpath / 'spi1_medwest_mean.nc')

"""
EC earth SLP and Z500, concatenate multiple members
but not multiple experiments? 
""" 

var = 'psl' # 'psl' or 'zg500'
index = 'ao'
ncvarname = {'psl':'psl','zg500':'zg'}[var]

domains = {'ao':{'lat':slice(20,None),'lon':slice(None,None)},
        'nao':{'lat':slice(20,None),'lon':slice(-90,30)}}

arrs = []
for amoc_str, shift in amoc_strs.items():
    files = np.sort(list(ecpath.glob(f'{var}_Amon*_{amoc_str}_*')))
    subarrs = []
    for i, filepath in enumerate(files):
        da, axis = process_one(filepath, ncvarname = ncvarname, nyearshift = shift)
        da = da.sel(**domains[index]) #{'lat':domains[index]['latslice'],'lon':domains[index]['lonslice']})
        if var == 'zg500':
            da = da.squeeze('plev').drop('plev')
        da = da.expand_dims({'amoc':[amoc_str],'member':[i]})
        subarrs.append(da)
    arrs.append(xr.concat(subarrs, 'member'))

total = xr.concat(arrs, 'amoc') 
anom = total.groupby(total.time.dt.month).apply(lambda a: a - a.mean(['time','member','amoc']))
#monmean = total.groupby(total.time.dt.month).mean(['time','member','amoc'])
#anom2 = total.groupby(total.time.dt.month) - monmean
superstack = anom.stack({'stackdim':['amoc','member','time']}) # season based subsetting?

model_lat = EOF(n_modes=2, standardize=False, use_coslat=True)
model_lat.fit(superstack, "stackdim")

outpath = datapath / 'ecearth'

loading_patterns = model_lat.components()
loading_patterns.attrs.pop('solver_kwargs') 
scores = model_lat.scores().unstack('stackdim')
scores.attrs.pop('solver_kwargs') 
if loading_patterns.sel(mode = 1).mean('lon').sel(lat = slice(35,65)).diff('lat').mean().values > 0: # Make sure that we have positive AO and positive NAO meaning that pressure closer to pole is less than normal
    loading_patterns.loc[1] = -loading_patterns.loc[1] 
    scores.loc[1] = -scores.loc[1] 
#loading_patterns.to_netcdf(outpath / f'{index}_{var}_components.nc')
#scores.to_netcdf(outpath / f'{index}_{var}_timeseries.nc')

"""
Gibraltar - Stykkisholmur definition
Jones, P.D., Jónsson, T. and Wheeler, D., 1997: Extension to the North Atlantic Oscillation using early instrumental pressure observations from Gibraltar and South-West Iceland. Int. J. Climatol. 17, 1433-1450
"""
if (index == 'nao') and (var == 'psl'):
    iceland = anom.sel(lat = 65,lon = 22.8, method = 'nearest')
    gibraltar = anom.sel(lat =36.1106, lon = 5.3466, method = 'nearest')
    station_based = gibraltar - iceland 
    station_based.to_netcdf(outpath / f'{index}_{var}_station_timeseries.nc')


"""
EC earth 250 winds, concatenate multiple members
no need for multiple experiments? 
""" 


