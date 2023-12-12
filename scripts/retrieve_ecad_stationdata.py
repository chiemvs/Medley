import sys
import os
import zarr
import fsspec
import warnings
import xarray as xr
import numpy as np
import pandas as pd
import pyarrow as pa

from pathlib import Path

sys.path.append(os.path.expanduser('~/Documents/Medley/'))
from Medley.utils import chunk_func, process_ascii, coord_to_decimal_coord
from Medley.preprocessing import monthly_resample_func

overwrite = True

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
monthly values (SPI includes the month itself, and previous months, depending on the aggregation)
"""

rawdir = Path(os.path.expanduser('~/Medi/monthly/eca_preaggregated_raw'))
#variables = {'CDD':'days','PET':'mm','SPI3':'','RR':'mm'}
#variables = {'RR':'mm'}
variables = {'SPI1_Gerard':''}

countries = ['SPAIN', 'PORTUGAL','FRANCE','ITALY','GREECE','ISRAEL','SLOVENIA','CROATIA','CYPRUS','MONTENEGRO','ALBANIA','NORTH MACEDONIA','BOSNIA AND HERZEGOVINA','TUNISIA']#,'TÃ\x9cRKIYE']
# Turkey is weirdly encoded

miss = -999999

def read_one_file(path):
    """
    reading and numerical processing in read_ascii
    conversion from units 0.01 to units 1
    01-06 SOUID: Source identifier
    08-11 YEAR : YYYY
    13-20 ANNUAL DATA VALUES
    21-28 WINTER HALF YEAR DATA VALUES
    29-36 SUMMER HALF YEAR DATA VALUES
    37-44 WINTER (DJF) DATA VALUES
    45-52 SPRING (MAM) DATA VALUES
    53-60 SUMMER (JJA) DATA VALUES
    61-68 AUTUMN (SON) DATA VALUES
    69-76 JANUARY DATA VALUES
    etc. 
    157-164 DECEMBER DATA VALUES
    """
    temp = pd.read_fwf(path, skiprows = 29, header = None, colspecs = [(7,11),(68,76),(76,84),(84,92),(92,100),(100,108),(108,116),(116,124),(124,132),(132,140),(140,148),(148,156),(156,164)])
    year = temp.iloc[:,0]
    jan_to_dec = temp.iloc[:,-12:]
    test = process_ascii(year.astype(np.float32), jan_to_dec.values.flatten(), miss_val = miss)
    return test / 100

def read_one_spi1_file(path):
    temp = pd.read_csv(path, header = None, names = ['ignore','STAID','month','year','SPI1','filter'])
    temp.loc[:,'month'] = temp.loc[:,'month'] - 6 # 0=jaarlijks gemiddelde, 1,2 winter- en zomerhalfjaar, 3,4,5,6 DJF, MAM, JJA, SON, 7=januari, 8=februari etc
    temp = temp.loc[temp.loc[:,'month'] > 0,:]
    decimal_year = temp.loc[:,'year'].astype(np.float32) + (temp.loc[:,'month'] - 1) / 12
    test = process_ascii(decimal_year.values, temp['SPI1'].values.flatten(), miss_val = miss)
    return test / 100


for var in variables.keys():
    zipname = f'ECA_index{var}.zip'
    zippath = rawdir / zipname
    tempvardir = rawdir / f'temp_{var}'
    if (not zippath.exists()) or overwrite:
        if var == 'SPI1_Gerard':
            warnings.warn('SPI1 cannot be downloaded, as was manually prepared by Gerard. Continuing with existing data')
        else:
            url = f'https://knmi-ecad-assets-prd.s3.amazonaws.com/download/millennium/data/{zipname}'
            os.system(f'curl {url} -o {zippath}')
            os.system(f'unzip {zippath} stations.txt -d {tempvardir}')
        stations = pd.read_fwf(f'{tempvardir}/stations.txt',
                colspecs = [(0,5),(6,46),(47,87),(88,97),(98,108),(109,114)], skiprows = 1,header = 11,encoding = 'latin')
        subset = stations.loc[stations['COUNTRYNAME'].apply(lambda n: (n in countries) or n.endswith('KIYE')).values,:]
        # Some reformatting
        subset.loc[:,'LAT'] = subset.loc[:,'LAT'].apply(coord_to_decimal_coord)
        subset.loc[:,'LON'] = subset.loc[:,'LON'].apply(coord_to_decimal_coord)
        subset = subset.astype({'LAT':np.float64,'LON':np.float64})
        subset.loc[subset['COUNTRYNAME'].apply(lambda n: n.endswith('KIYE')).values,'COUNTRYNAME'] = 'TURKEY'
        subset = subset.set_index('STAID')

        if var == 'SPI1_Gerard':
            potential_stations = {sid:tempvardir/ f'index_SPI-1_{sid}.txt' for sid in subset.index}
            data = {sid: read_one_spi1_file(path) for sid, path in potential_stations.items() if path.exists()}
            subset = subset.loc[data.keys(),:] # Not everything is prepared by Gerard
        else:
            to_extract = {sid:f'index{var}' + f'000000{sid}'[-6:] + '.txt' for sid in subset.index}
            os.system(f'unzip {zippath} {" ".join(to_extract.values())} -d {tempvardir}')
            data = {sid: read_one_file(tempvardir / filename) for sid, filename in to_extract.items()}
        dataframe = pd.concat(data, axis = 1, join = 'outer')
        dataframe.columns = dataframe.columns.droplevel(-1).set_names('STAID')

        newdatapath = rawdir.parent / f'eca_preaggregated_{var}.h5'
        stationpath = rawdir.parent / f'eca_preaggregated_{var}_stations.h5'

        dataframe.to_hdf(newdatapath, key = var, mode = 'w')
        subset.to_hdf(stationpath, key = var, mode = 'w')
