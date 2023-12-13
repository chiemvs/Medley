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

sys.path.append(os.path.expanduser('~/Documents/Medley/'))
from Medley.utils import udomains, tscolnames

datapath = Path('/scistor/ivm/jsn295/Medi/monthly')

def load_climexp() -> pd.DataFrame:
    """
    Collecting all the ready timeseries data from climexp
    previously saved to disk with the retrieve climexp script
    """
    inpath = datapath / f'monthly_climexp.h5'
    df = pd.read_hdf(inpath)
    return df

def u_series(name, lonmin, lonmax, level = 250) -> pd.DataFrame:
    """
    Monthly mean zonal mean u250
    Separate for subtropical jet and north atlantic. 
    Extracting latitude of maximum, and strengths at
    40N and 55N
    """
    upath = datapath / 'era5' / f'monthly_zonalmean_u{level}_NH_{lonmin}E_{lonmax}E.nc'
    uda = xr.open_dataarray(upath)
    latpath = datapath / 'era5' / f'monthly_zonallatmax_u{level}_NH_{lonmin}E_{lonmax}E.nc'
    latda = xr.open_dataarray(latpath)
    collection = latda.to_dataframe()
    collection.columns = pd.MultiIndex.from_tuples([(f'{name}_u{level}_latmax',0,'era5')], names = tscolnames)
    for lat in range(20,70,10):
        ts = uda.sel(latitude = lat, drop = True).to_dataframe().astype(np.float32)
        ts.columns = pd.MultiIndex.from_tuples([(f'{name}_u{level}',lat,'era5')], names = tscolnames)
        collection = collection.join(ts, how = 'left') # Indices should match so how is irrelevant.
    return collection

def all_u_series():
    collection = []
    for level in [250,500]:
        for name, (lonmin, lonmax) in udomains.items():
            u = u_series(name = name, lonmin = lonmin, lonmax = lonmax, level = level)
            collection.append(u) 
    return pd.concat(collection, axis = 1)

def vortex():
    """
    Definition of Zappa and Shepherd 2017
    """
    vortexpath = datapath / 'era5' / 'monthly_u20_era5.zarr'
    uwinds = xr.open_zarr(vortexpath)
    uwinds = uwinds.sel(latitude = slice(80,70)).mean(['longitude','latitude'])
    ts = uwinds.to_dataframe() # invokes the compute
    ts.columns = pd.MultiIndex.from_tuples([('vortex_u20',7080,'era5')], names = tscolnames)
    return ts

def indian_ocean():
    """
    Indian ocean dipole and its east and west constituents
    western equatorial Indian Ocean (50E-70E and 10S-10N) 
    south eastern equatorial Indian Ocean (90E-110E and 10S-0N). 
    difference (west-east) is IOD or Dipole Mode Index (DMI)
    """
    iodspath = datapath / 'monthly_iods.h5'
    return pd.read_hdf(iodspath, key = 'iods')

def amoc_proxy():
    """
    based on hadisst fingerprint of:
    Caesar, L., Rahmstorf, S., Robinson, A., Feulner, G., & Saba, V. (2018). Observed fingerprint of a weakening Atlantic Ocean overturning circulation. Nature, 556(7700), 191-196.
    """
    amocpath = datapath / 'monthly_amoc_proxy.h5' 
    return pd.read_hdf(amocpath, key = 'amoc')

def eke_series():
    """
    Only the atlantic domain
    """
    lonmin, lonmax = udomains['atl']
    path = datapath / f'monthly_zonalmean_EKE_NH_{lonmin}E_{lonmax}E.nc'
    da = xr.open_dataarray(path)
    collection = da.idxmax('latitude').to_dataframe().unstack('Reanalysis').astype(np.float32)
    collection.columns = pd.MultiIndex.from_product([('atl_eke_latmax',),(0,),collection.columns.get_level_values('Reanalysis')], names = tscolnames)
    for lat in range(20,70,10):
        ts = da.sel(latitude = lat, drop = True, method = 'nearest').to_dataframe().unstack('Reanalysis').astype(np.float32)
        ts.columns = pd.MultiIndex.from_product([('atl_eke',),(lat,),ts.columns.get_level_values('Reanalysis')], names = tscolnames)
        collection = collection.join(ts, how = 'left') # Indices should match so how is irrelevant.
    # For now only ERA5
    return collection.loc[:,(slice(None),slice(None),['ERA5'])]


def make_monthly_data(force_update: bool = False):
    """
    Main function, reads data if stored
    else will run the downloading and processing (or when update is forced).
    """
    finalpath = datapath / 'complete.parquet'
    if (not finalpath.exists()) or force_update: # Possibly need to unlink for updating
        if finalpath.exists():
            os.system(f'cp {finalpath} {finalpath}.backup')
        climexp_df = load_climexp() # download_climexp()
        era_df = all_u_series() # jet stream u's
        vortex_df = vortex() # stratospheric vortex u's
        eke_df = eke_series() 
        iod_df = indian_ocean()
        amoc_df = amoc_proxy()
        complete = climexp_df.join(era_df, how = 'outer') # Should not add rows to climexp_df as that is much larger.
        complete = complete.join(vortex_df, how = 'outer')
        complete = complete.join(eke_df, how = 'outer')
        complete = complete.join(iod_df, how = 'outer')
        complete = complete.join(amoc_df, how = 'outer')
        table = pa.Table.from_pandas(complete) # integer multiindex level becomes text
        pq.write_table(table, finalpath)
    else:
        table = pq.read_table(finalpath)
    return table

if __name__ == '__main__':
    #test = u_series(name='atlantic', lonmin = -50, lonmax = -10, level = 500)
    #test2 = u_series(name='atlantic', lonmin = -8.5, lonmax = 42, level = 500)
    df = make_monthly_data(force_update = False).to_pandas()
