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
from Medley.utils import chunk_func, process_ascii
from Medley.preprocessing import monthly_resample_func

overwrite = True
blend = True

"""
Daily data
"""
var = 'rr'

#url = f'https://knmi-ecad-assets-prd.s3.amazonaws.com/download/ECA_{"" if blend else "non"}blend_{variable}.zip'

tempdir = Path('/scistor/ivm/jsn295/temp')

#os.system(f'curl {url} -o {tempdir / "temp.zip"}')
#os.system(f'unzip {tempdir / "temp.zip"} -d {tempdir}')

for country_zip in tempdir.glob(f'ECA_blended_{var}_*.zip'):
    country = country_zip.name.split('_')[-1].split('.')[0] 

"""
Predefined indices
monthly values
"""

rawdir = Path(os.path.expanduser('~/Medi/monthly/eca_preaggregated_raw'))
variables = {'CDD':'days','PET':'mm','SPI3':'','RR':'mm'}
variables = {'RR':'mm'}

countries = ['SPAIN', 'PORTUGAL','FRANCE','ITALY','GREECE','ISRAEL','SLOVENIA','CROATIA','CYPRUS','MONTENEGRO','ALBANIA']#,'TÃ\x9cRKIYE']
# Turkey is weirdly encoded

miss = -999999.0

def read_one_file(path):
    """
    reading and numerical processing in read_ascii
    conversion from units 0.01 to units 1
    """
    temp = pd.read_fwf(path, skiprows = 29, header = None) 
    year = temp.iloc[:,1]
    jan_to_dec = temp.iloc[:,-12:]
    test = process_ascii(year.astype(np.float32), jan_to_dec.astype(np.float32).values.flatten(), miss_val = miss)
    return test / 100

for var in variables.keys():
    zipname = f'ECA_index{var}.zip'
    zippath = rawdir / zipname
    tempvardir = rawdir / f'temp_{var}'
    if (not zippath.exists()) or overwrite:
        url = f'https://knmi-ecad-assets-prd.s3.amazonaws.com/download/millennium/data/{zipname}'
        #os.system(f'curl {url} -o {zippath}')
        #os.system(f'unzip {zippath} stations.txt -d {tempvardir}')
        stations = pd.read_fwf(f'{tempvardir}/stations.txt',
                colspecs = [(0,5),(6,46),(47,87),(88,97),(98,108),(109,114)], skiprows = 1,header = 11,encoding = 'latin')
        subset = stations.loc[stations['COUNTRYNAME'].apply(lambda n: (n in countries) or n.endswith('KIYE')).values,:]

        to_extract = {sid:f'index{var}' + f'000000{sid}'[-6:] + '.txt' for sid in subset['STAID']}
        #paths = {sid:tempvardir / to_extract[sid] for sid in to_extract.keys()}

        #os.system(f'unzip {zippath} {" ".join(to_extract.values())} -d {tempvardir}')
        data = {sid: read_one_file(tempvardir / filename) for sid, filename in to_extract.items()}
        temp = pd.concat(data, axis = 1, join = 'outer')
