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
from scipy.interpolate import interp1d
from typing import Union

sys.path.append(os.path.expanduser('~/Documents/Medley/'))
from Medley.utils import process_ascii, tscolnames

"""
Caesar, L., Rahmstorf, S., Robinson, A., Feulner, G., & Saba, V. (2018). Observed fingerprint of a weakening Atlantic Ocean overturning circulation. Nature, 556(7700), 191-196.

Caesar, L., McCarthy, G. D., Thornalley, D. J. R., Cahill, N., & Rahmstorf, S. (2021). Current Atlantic meridional overturning circulation weakest in last millennium. Nature Geoscience, 14(3), 118-120.
"""

datapath = Path('/scistor/ivm/jsn295/Medi/')
rawdatapath = datapath / 'annual'/ 'amoc_raw_data' 
finalpath = datapath / 'monthly'/ 'monthly_amoc_proxy.h5' 

all_indices = 'https://raw.githubusercontent.com/ncahill89/AMOC-Analysis/main/data/Data_compilation.xlsx'
one_only = 'https://raw.githubusercontent.com/ncahill89/AMOC-Analysis/main/data/Caesar.csv' 



def interpolate(inp: Union[pd.DataFrame, pd.Series], kind: str = 'linear'):
    """
    Interpolation to monthly values of all columns in a dataframe
    """
    if isinstance(inp, pd.Series):
        convert = True
        inp = inp.to_frame()
    else:
        convert = False
    interpolated = pd.DataFrame(np.nan, index = pd.date_range(inp.index.values.min(), inp.index.values.max(), freq = 'MS'), columns = inp.columns)
    for key in inp.columns:
        f = interp1d(inp.index.to_julian_date(), inp.loc[:,key].values, kind = kind)
        interpolated.loc[:,key] = f(interpolated.index.to_julian_date())
    if convert:
        return interpolated.iloc[:,-1] # Returning as series again
    else:
        return interpolated


if __name__ == '__main__':
    overwrite = False
    #os.system(f'curl {all_indices} -o {rawdatapath/"amoc_indices.xlsx"}')
    #os.system(f'curl {one_only} -o {rawdatapath/"ceasar.csv"}')
    
    if (not finalpath.exists()) or overwrite:
        df = pd.read_csv(rawdatapath/"ceasar.csv", index_col = 0)
        df.index = pd.to_datetime(df.index, format = '%Y') + pd.Timedelta(181, 'd') # Stamp in the middle of the year
        out = interpolate(df)[['y']]
        out.columns = pd.MultiIndex.from_tuples([('amoc',0,'hadisst')], names = tscolnames)

        out.to_hdf(finalpath, key = 'amoc', mode = 'w')


