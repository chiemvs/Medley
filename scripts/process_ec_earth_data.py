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
from Medley.utils import transform_to_spi, regions, udomains
from Medley.preprocessing import monthly_resample_func, makemask
from Medley.dataloading import datapath

"""
This is a relatively quick-and-dirty script that is not part of the analysis of observational data
and neither of the statistical forecasting pipeline.
Instead, these are simulation data, described in https://iopscience.iop.org/article/10.1088/1748-9326/ad14b0/pdf
Several variables (an expert selection) are stored in IVM's data_catalogue
Here these are processed with the goal of producing timeseries for a causal analysis and not much more.
Definitions for Arctic Oscillation and North Atlantic Oscillation are crude.
"""

def process_one(filepath, ncvarname: str, nyearshift: int = 0):
    """
    Dropping first year for independence of the atmospheric runs
    nyearshift for concatenation, monthly timeaxis will be left-stamped
    making sure longitude is increasing and -180 to 180
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

outpath = datapath / 'ecearth'

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
#spi.unstack('latlon').sortby('lat').sortby('lon').to_netcdf(outpath / 'spi1_medwest.nc')
#spi.mean('latlon').to_netcdf(outpath / 'spi1_medwest_mean.nc')

"""
EC earth SLP and Z500, concatenate multiple members
but not multiple experiments? 
""" 

var = 'zg500' # 'psl' or 'zg500'
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

def reduce_to_two_pcs(array: xr.DataArray):
    """
    Stacked 2D array such that the stackdim contains samples
    """
    model_lat = EOF(n_modes=2, standardize=False, use_coslat=True)
    model_lat.fit(array, "stackdim")
    
    loading_patterns = model_lat.components()
    loading_patterns.attrs.pop('solver_kwargs') 
    scores = model_lat.scores().unstack('stackdim')
    scores.attrs.pop('solver_kwargs') 
    if loading_patterns.sel(mode = 1).mean('lon').sel(lat = slice(35,65)).diff('lat').mean().values > 0: # Make sure that we have positive AO and positive NAO meaning that pressure closer to pole is less than normal
        loading_patterns.loc[1] = -loading_patterns.loc[1] 
        scores.loc[1] = -scores.loc[1] 
    return loading_patterns, scores, model_lat

"""
Version with all data at once
"""
#loading_patterns, scores, _ = reduce_to_two_pcs(superstack)
#loading_patterns.to_netcdf(outpath / f'{index}_{var}_components.nc')
#scores.to_netcdf(outpath / f'{index}_{var}_timeseries.nc')

"""
Version whereby current climate EOF is projected on the reduced strength data
"""
if (index == 'ao') and (var == 'zg500'): 
    stack_17 = anom.sel(amoc = '17sv').stack({'stackdim':['member','time']})
    stack_14 = anom.sel(amoc = '14sv').stack({'stackdim':['member','time']})
    stack_07 = anom.sel(amoc = '07sv').stack({'stackdim':['member','time']})
    loading_patterns_17, scores_17, model_17 = reduce_to_two_pcs(stack_17)
    loading_patterns_17.drop('amoc').to_netcdf(outpath / f'{index}_{var}_17sv_components.nc')
    scores_14 = model_17.transform(stack_14).unstack('stackdim')
    scores_07 = model_17.transform(stack_07).unstack('stackdim')
    scores = xr.concat([scores_17,scores_14,scores_07], dim = pd.Index(['17sv','14sv','7sv'], name = 'amoc'))
    scores.to_netcdf(outpath / f'{index}_{var}_17sv_timeseries.nc')


"""
Gibraltar - Stykkisholmur definition
Jones, P.D., Jónsson, T. and Wheeler, D., 1997: Extension to the North Atlantic Oscillation using early instrumental pressure observations from Gibraltar and South-West Iceland. Int. J. Climatol. 17, 1433-1450
"""
#if (index == 'nao') and (var == 'psl'):
#    iceland = anom.sel(lat = 65,lon = 22.8, method = 'nearest')
#    gibraltar = anom.sel(lat =36.1106, lon = 5.3466, method = 'nearest')
#    station_based = gibraltar - iceland 
#    station_based.to_netcdf(outpath / f'{index}_{var}_station_timeseries.nc')


"""
EC earth 250 zonal (u) winds
zonal mean of maximal latitude, and speed of zonal mean at multiple latitudes
(latitudes closest to 20,30,40,50,60 N.)
computed for two domains: mediterranean and atlantic
""" 
#
#for lonmin, lonmax in udomains.values():
#    lonslice = slice(lonmin,lonmax,None)
#    latslice = slice(0, None, None)  # stored with Decreasing latitude
#    arrs = []
#    for amoc_str, shift in amoc_strs.items():
#        files = np.sort(list(ecpath.glob(f'ua250_Amon*_{amoc_str}_*')))
#        subarrs = []
#        for i, filepath in enumerate(files):
#            da, axis = process_one(filepath, ncvarname = 'ua', nyearshift = shift)
#            da = da.sel({'lat':latslice,'lon':lonslice})
#            da = da.squeeze('plev').drop('plev')
#            da = da.expand_dims({'amoc':[amoc_str],'member':[i]})
#            subarrs.append(da)
#        arrs.append(xr.concat(subarrs, 'member'))
#    
#    total = xr.concat(arrs, 'amoc') 
#
#    # To timeseries
#    zonal_u_mean = total.mean('lon').sel(lat=list(range(20,70,10)),method = 'nearest')
#    zonal_u_mean.attrs = da.attrs
#    zonal_lat_mean = total.idxmax('lat').mean('lon') # Order presented by Albert Osso
#    zonal_lat_mean.attrs.update({'standard_name':'zonalmean_latmax','long_name':'latitude_of_maximum_per_longitude_then_zonal_mean'})
#    zonal_u_mean.to_netcdf( outpath / f'monthly_zonalmean_u250_NH_{lonmin}E_{lonmax}E.nc')
#    zonal_lat_mean.to_netcdf( outpath / f'monthly_zonallatmax_u250_NH_{lonmin}E_{lonmax}E.nc')

"""
EC earth 20 Hpa winds
""" 
#arrs = []
#for amoc_str, shift in amoc_strs.items():
#    files = np.sort(list(ecpath.glob(f'ua20_Amon*_{amoc_str}_*')))
#    subarrs = []
#    for i, filepath in enumerate(files):
#        da, axis = process_one(filepath, ncvarname = 'ua', nyearshift = shift)
#        da = da.sel({'lat':slice(70,80)})
#        da = da.squeeze('plev').drop('plev')
#        da = da.expand_dims({'amoc':[amoc_str],'member':[i]})
#        subarrs.append(da)
#    arrs.append(xr.concat(subarrs, 'member'))
#
#total = xr.concat(arrs, 'amoc') 
#
## To timeseries
#vortex = total.mean(['lat','lon'])
#vortex.attrs = da.attrs
#vortex.to_netcdf( outpath / f'monthly_zonalmean_u20_NH_70N_80N.nc')


