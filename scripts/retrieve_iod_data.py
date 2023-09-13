
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
from Medley.utils import process_ascii, tscolnames

datapath = Path('/scistor/ivm/jsn295/Medi/monthly')
rawdatapath = datapath / 'iod_raw_data' 

"""
https://psl.noaa.gov/gcos_wgsp/Timeseries/DMI/
Indian ocean dipole and its east and west constituents
western equatorial Indian Ocean (50E-70E and 10S-10N) 
south eastern equatorial Indian Ocean (90E-110E and 10S-0N). 
difference (west-east) is IOD or Dipole Mode Index (DMI)
"""
dataproperties = pd.DataFrame({'iod':['iod',0,'hadisst','https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data'],
    'west':['westind',5070,'hadisst','https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmiwest.had.long.data'],
    'east':['eastind',90110,'hadisst','https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmieast.had.long.data'],
    },
    index = tscolnames + ['url']).T

def download_and_load_data(name:str, overwrite = False) -> tuple[array.array,array.array]:
    """
    Writing raw file including metadata
    reading only the non-metadata
    """
    localfile = rawdatapath / f'{name}.dat' 
    if (not localfile.exists()) or overwrite:
        cmd = f'curl {dataproperties.loc[name,"url"]} -o {localfile}'
        os.system(cmd)
    #data = pd.read_fwf(localfile, index_col = 0, infer_nrows = 100, skiprows = 0, skipfooter = 7, header = None) # This fails to get the correct missval
    timestamps = array.array('f') # Appendable formats
    values = array.array('f')
    with open(localfile) as f:
        for line in f:
            content = line.splitlines()[0] # Can lead to an empty string if only \n
            if len(content) > 70: # Short lines are metadata
                content = [float(s) for s in content.split(' ') if s] 
                timestamps.append(content.pop(0))
                values.extend(content)
    return timestamps, values

if __name__ == '__main__':
    overwrite = False
    combined_file = datapath / 'monthly_iods.h5'
    if (not combined_file.exists()) or overwrite:
        combined = []
        for name in dataproperties.index:
            #content = download_and_load_data(name = name)
            timestamps, values = download_and_load_data(name = name, overwrite = overwrite)
            data = process_ascii(timestamps, values, miss_val = -9999)
            data.columns = pd.MultiIndex.from_frame(dataproperties.loc[[name],tscolnames])
            combined.append(data)
        combined = pd.concat(combined, axis = 1)
        combined.to_hdf(combined_file, key = 'iods', mode = 'w')
