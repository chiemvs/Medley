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
Script to put everything into zarr stores
ensemble means of EOBS
"""

overwrite = False
variable = 'rr'
version = '27.0e'
resolution = 0.1 

ncname = f'{variable}_ens_mean_{resolution}deg_reg_v{version}.nc'
onlinepath = f'https://knmi-ecad-assets-prd.s3.amazonaws.com/ensembles/data/Grid_{resolution}deg_reg_ensemble/' + ncname
path_for_everyone = Path('/scistor/ivm/data_catalogue/observations/EOBS') / ncname 

def chunk_and_to_zarr(ncpath: Path, varname: str , to_monthly: bool = False) -> tuple[xr.DataArray,Path]:
    """ Returns array and suggested zarrpath """
    assert ncpath.exists(), 'please provide an existing path'
    zarrpath = ncpath.parent / (ncpath.parts[-1][:-3] + '.zarr') # Stripping .nc and replacing with zarr
    daskarray = xr.open_dataset(ncpath).drop_vars('time_bnds', errors = 'ignore') # Keeping as dataset.
    if to_monthly:
        daskarray = monthly_resample_func(daskarray, how = 'sum')
        daskarray = chunk_func(daskarray)
    # Writing and filling
    return daskarray, zarrpath

if (not path_for_everyone.exists()) or overwrite:
    os.system(f'curl {onlinepath} -o {path_for_everyone}')

"""
Resampling from daily to monthly
"""
datadir = Path('/scistor/ivm/jsn295/Medi/monthly/')
daskarray, zarrpath = chunk_and_to_zarr(ncpath = path_for_everyone,
                                     varname = 'rr', to_monthly = True)
zarrpath = datadir / f'{zarrpath.name[:3]}mon_{zarrpath.name[3:]}'
if (not zarrpath.exists()) or overwrite:
    daskarray.to_zarr(zarrpath, consolidated=True) # adding month to name
