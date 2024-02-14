import zarr
import lilio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import xarray as xr

from typing import Union
from pathlib import Path

from .utils import tscolnames, udomains
from .preprocessing import makemask, average_within_mask, single_target_lagged_resample, multi_target_lagged_resample 

"""
IMPORTANT! set the below path to where all monthly data is stored.
"""
datapath = Path('/scistor/ivm/jsn295/Medi/monthly/')

def get_monthly_data():
    """
    Loading the monthly data prepared by 'make_monthly_data' in the scripts/prepare_monthly_ts_data
    If you want to update, rerun that script.
    """
    finalpath = datapath / 'complete.parquet'
    table = pq.read_table(finalpath)
    return table.to_pandas()

def prep_ecad(target_region: dict, target_var: str = 'RR', minsamples: int = 10, shift: bool = False) -> pd.Series:
    """
    Loading ECAD data
    target region dictionary, see Medley.utils.regions 
    possible to apply an (X-1) month shift when SPIX is the target_var
    SPI-3 for instance has a three month accumulation, timestamped at the last month
    We should shift it with two months to achieve independence.
    For SPI-1 it does not matter.
    """
    mask = makemask(target_region) 
    ecad = pd.read_hdf(datapath / f'eca_preaggregated_{target_var}.h5')
    ecad.index.name = 'time' # Monthly DatetimeIndex
    if shift and target_var.startswith('SPI'):
        n_months = int(target_var[-1])
        ecad.index = pd.date_range(end = ecad.index[-n_months], periods = len(ecad.index), freq = 'MS')
    ecad_locs = pd.read_hdf(datapath / f'eca_preaggregated_{target_var}_stations.h5')
    target = average_within_mask(mask = mask, data = ecad, datalocs = ecad_locs, minsamples=minsamples)
    return target

def prep_and_resample(target_region: dict, target_var: str = 'RR', minsamples: int = 10, shift: bool = False, resampling : str = 'single', resampling_kwargs : dict = {}) -> tuple[pd.DataFrame, pd.DataFrame, lilio.Calendar]:
    """
    Bulk function capturing common data preparation steps (both X and y)
    Calls the Loading ECAD data (creating a y)
    target region dictionary, see Medley.utils.regions 
    Calls the get_monthly_data function (creating an X)
    Then applies resampling and lagging.
    kwargs fed to the respective resampling function
    """
    target = prep_ecad(target_region, target_var, minsamples, shift).to_frame()
    target.columns = pd.MultiIndex.from_tuples([(target_var,0,'ECAD')], names = tscolnames)
    df = get_monthly_data()

    # Define temporal sampling approaches
    if resampling == 'single':
        try:
            resampling_kwargs.pop('target_agg') # kwarg ignored in single resampling.
        except KeyError:
            pass
        Xm, ym, cm = single_target_lagged_resample(X = df, y = target, **resampling_kwargs) 
    elif resampling == 'multi':
        Xm, ym, cm = multi_target_lagged_resample(X = df, y = target, **resampling_kwargs) 
    else:
        raise ValueError('invalid resampling instruction given, should be one of "multi" or "single"')
    return Xm, ym, cm

def load_ecearth_var(name: str, subindex: int, product: str, filename: str, fake_time_axis: bool = True) -> pd.DataFrame:
    """
    load of a single ec-earth variable
    transforming the array to usable dataframe 
    with same time axis, and also indexed by amoc strength
    """
    path = datapath / 'ecearth' / filename
    da = xr.open_dataarray(path)
    if name.endswith('u250'):
        da = da.sel(lat=subindex,method = 'nearest')
    if 'mode' in da.dims: # Is the case for NAO and AO
        da = da.sel(mode = 1)
    df = da.stack({'mt':['member','time']}).T.to_pandas()
    # Resetting time axes here
    if fake_time_axis:
        newindex = df.index.to_frame()
        members = df.index.get_level_values('member').unique()
        nyears = 11 # per member timeseries, 10 plus 1 for the gap
        for m in members:
            newindex.loc[(m,slice(None)),'time'] = pd.date_range(f'{1700+m*(nyears)}-01-01',f'{1709+m*(nyears)}-12-31', freq = 'MS')
        df.index = pd.MultiIndex.from_frame(newindex)
    return df

def prepare_ecearth_set(fake_time_axis: bool = True) -> pd.DataFrame:
    """
    Loading of all variables. (hardcoded through filenames which ones) 
    option for extending time axis when joining members (to fit everything into same frame).
    If not, then member needs to become a context variable (just like AMOC), also in the dataframe itself
    """
    productname = 'ecearth'
    path_names = {}
    for name, (lonmin, lonmax) in udomains.items():
        for lat in range(20,70,10):
            path_names.update({(f'{name}_u250',lat,productname): f'monthly_zonalmean_u250_NH_{lonmin}E_{lonmax}E.nc'})
        path_names.update({(f'{name}_u250_latmax',0,productname): f'monthly_zonallatmax_u250_NH_{lonmin}E_{lonmax}E.nc'})

    path_names.update({('vortex_u20',7080,productname):'monthly_zonalmean_u20_NH_70N_80N.nc'})
    path_names.update({('ao',0,productname):'ao_psl_timeseries.nc'})
    path_names.update({('nao',0,productname):'nao_psl_timeseries.nc'})
    path_names.update({('nao',0,'station'):'nao_psl_station_timeseries.nc'})
    path_names.update({('SPI1',0,productname):'spi1_medwest_mean.nc'})

    loaded_sets = {}
    for keytup, filename in path_names.items():
        loaded_sets[keytup] = load_ecearth_var(*keytup, filename = filename, fake_time_axis = fake_time_axis) 
    result = pd.concat(loaded_sets, axis = 1)
    result.columns = result.columns.set_names(tscolnames + ['amoc'])
    return result

