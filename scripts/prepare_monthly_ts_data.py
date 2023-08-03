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
    path = datapath / f'monthly_zonalmean_u{level}_NH_{lonmin}E_{lonmax}E.nc'
    da = xr.open_dataarray(path)
    collection = da.idxmax('latitude').to_dataframe()
    collection.columns = pd.MultiIndex.from_tuples([(f'{name}_u{level}_latmax',0,'era5')], names = tscolnames)
    for lat in range(20,70,10):
        ts = da.sel(latitude = lat, drop = True).to_dataframe().astype(np.float32)
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
    vortexpath = datapath / 'monthly_u20_era5.zarr'
    uwinds = xr.open_zarr(vortexpath)
    uwinds = uwinds.sel(latitude = slice(80,70)).mean(['longitude','latitude'])
    ts = uwinds.to_dataframe() # invokes the compute
    ts.columns = pd.MultiIndex.from_tuples([('vortex_u20',7080,'era5')], names = tscolnames)
    return ts

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
        complete = climexp_df.join(era_df, how = 'outer') # Should not add rows to climexp_df as that is much larger.
        complete = complete.join(vortex_df, how = 'outer')
        complete = complete.join(eke_df, how = 'outer')
        table = pa.Table.from_pandas(complete) # integer multiindex level becomes text

        pq.write_table(table, finalpath)
    else:
        table = pq.read_table(finalpath)
    return table

if __name__ == '__main__':
    df = make_monthly_data(force_update = False).to_pandas()
    #collection = eke_series()
