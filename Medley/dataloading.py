import zarr
import lilio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import xarray as xr

from typing import Union
from pathlib import Path

from .utils import tscolnames
from .preprocessing import makemask, average_within_mask, single_target_lagged_resample, multi_target_lagged_resample 

datapath = Path('/scistor/ivm/jsn295/Medi/monthly/')

def get_monthly_data():
    """
    Loading the monthly data prepared by 'make_monthly_data' in the scripts
    """
    finalpath = datapath / 'complete.parquet'
    table = pq.read_table(finalpath)
    return table.to_pandas()

def prep_ecad(target_region: dict, target_var: str = 'RR', minsamples: int = 10, shift: bool = False) -> pd.Series:
    """
    target region dictionary
    Loading ECAD data
    possible to apply an (X-1) month shift when SPIX is the target_var
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
    target region dictionary
    Loading ECAD data
    resampling determines the resampling strategy
    kwargs fed to the respective resampling function
    """
    target = prep_ecad(target_region, target_var, minsamples, shift).to_frame()
    target.columns = pd.MultiIndex.from_tuples([(target_var,0,'ECAD')], names = tscolnames)
    df = get_monthly_data()

    # Define temporal sampling approaches
    if resampling == 'single':
        try:
            resampling_kwargs.pop('target_agg')
        except KeyError:
            pass
        Xm, ym, cm = single_target_lagged_resample(X = df, y = target, **resampling_kwargs) 
    elif resampling == 'multi':
        Xm, ym, cm = multi_target_lagged_resample(X = df, y = target, **resampling_kwargs) 
    else:
        raise ValueError('invalid resampling instruction given, should be one of "multi" or "single"')
    return Xm, ym, cm
