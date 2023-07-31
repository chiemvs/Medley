
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

climexp_properties = pd.DataFrame({
    'ersst_nino_12':['enso',12,'ersstv5','iersst_nino12a_rel.dat'],
    'ersst_nino_3':['enso',3,'ersstv5','iersst_nino3a_rel.dat'],
    'ersst_nino_34':['enso',34,'ersstv5','iersst_nino3.4a_rel.dat'], # Relative nino's
    'ersst_nino_4':['enso',4,'ersstv5','iersst_nino4a_rel.dat'],
    'ersst_amo':['amo',0,'ersstv5','iamo_ersst.dat'],
    'hadsst_amo':['amo',0,'hadsst4','iamo_hadsst.dat'],
    'rapid_amoc':['amoc',0,'rapid','imoc_mar_hc10_mon.dat'],
    'cpc_nao':['nao',0,'ncep','icpc_nao.dat'],
    'stat_nao':['nao',0,'station','inao.dat'], # 12 times a monthly value, then an annual value. https://crudata.uea.ac.uk/cru/data/nao/
    'ncar_snao':['snao',0,'ncep','isnao_ucar.dat'],
    'cpc_ao':['ao',0,'ncep','icpc_ao.dat'], # HAs a two-part time index
    'slp_ao':['ao',0,'trenb','iao_slp_ext.dat'],
    'cpc_ea':['ea',0,'ncep','icpc_ea.dat'], # Broken link in climexp to metadata. Look here: https://www.cpc.ncep.noaa.gov/data/teledoc/ea.shtml
    #'hadsst_pdo':['pdo',0,'hadsst3','ipdo_hadsst3.dat'], # Has missing years
    'ersst_pdo':['pdo',0,'ersstv5','ipdo_ersst.dat'],
    'hadcrut_gmst':['gmst',0,'hadcrut5','ihadcrut5_global.dat'],
    'cpc_mjo3':['mjo',3,'ncep','icpc_mjo03_mean12.dat'],
    'cpc_mjo6':['mjo',6,'ncep','icpc_mjo06_mean12.dat'],
    'ncar_qbo':['qbo',0,'ncep','inqbo.dat'],
    }, index = tscolnames + ['climexpfile']) 

def download_raw_data(name:str) -> tuple[array.array,array.array]:
    """
    Only timeseries
    Writing metadata to file, reading numeric data into memory
    """
    filename = climexp_properties.loc['climexpfile',name]
    remote_file = fsspec.open(f'http://climexp.knmi.nl/data/{filename}', mode = 'rt')
    timestamps = array.array('f') # Appendable formats
    values = array.array('f')
    with remote_file as rf:
        with open(datapath / 'climexp_metadata' / f'{name}.metadata', mode = 'wt') as mf:
            for line in rf:
                if line.startswith('#'):
                    mf.write(line)
                else:
                    content = line.splitlines()[0] # Can lead to an empty string if only \n
                    if content:
                        content = [float(s) for s in content.split(' ') if s] 
                        if name == 'cpc_ao':
                            timestamps.append(content.pop(0) + (content.pop(0)-1)/12) # year and a month to decimal year
                        else:
                            timestamps.append(content.pop(0))
                        if name == 'stat_nao':
                            content.pop(-1) # Last one is an annual value
                        values.extend(content)
    return timestamps, values


def download_climexp() -> pd.DataFrame:
    """Collecting all the ready timeseries data from climexp"""
    collection = []
    for name in climexp_properties.keys():
        print(f'starting with: {name}')
        t1, v1 = download_raw_data(name)
        ts = process_ascii(t1, v1)
        ts.columns = pd.MultiIndex.from_tuples([tuple(climexp_properties[name].values)],names = climexp_properties.index)
        ts.columns = ts.columns.droplevel(-1)
        collection.append(ts)
    return pd.concat(collection, axis = 1, join = 'outer') 

if __name__ == '__main__':
    overwrite = False
    outpath = datapath / f'monthly_climexp.h5'
    if (not outpath.exists()) or overwrite:
        df = download_climexp()
        df.to_hdf(outpath, key = 'climexp', mode = 'w')
    else:
        print("file already existing while overwrite = False")
