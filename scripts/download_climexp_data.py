import os
import sys
import zarr
import array
import tempfile
import fsspec
import warnings
import numpy as np
import xarray as xr
import pandas as pd

from pathlib import Path
from datetime import datetime, timedelta

climexp_properties = pd.DataFrame({
    'ersst_nino_1':['enso',1,'ersstv5','iersst_nino3a_rel.dat'],
    'ersst_nino_2':['enso',2,'ersstv5','iersst_nino3a_rel.dat'],
    'ersst_nino_3':['enso',3,'ersstv5','iersst_nino3a_rel.dat'],
    'ersst_nino_34':['enso',34,'ersstv5','iersst_nino3.4a_rel.dat'], # Relative nino's
    'ersst_nino_4':['enso',4,'ersstv5','iersst_nino4a_rel.dat'],
    'ersst_amo':['amo',0,'ersstv5','iamo_ersst.dat'],
    'hadsst_amo':['amo',0,'hadsst4','iamo_hadsst.dat'],
    'rapid_amoc':['amoc',0,'rapid','imoc_mar_hc10_mon.dat'],
    'cpc_nao':['nao',0,'ncep','icpc_nao.dat'],
    'stat_nao':['nao',0,'station','inao.dat'],
    'ncar_snao':['snao',0,'ncep','isnao_ucar.dat'],
    'slp_ao':['ao',0,'ncep','iao_slp_ext.dat'],
    'cpc_ea':['ea',0,'ncep','icpc_ea.dat'],
    #'hadsst_pdo':['pdo',0,'hadsst3','ipdo_hadsst3.dat'], # Has missing years
    'ersst_pdo':['pdo',0,'ersstv5','ipdo_ersst.dat'],
    'hadcrut_gmst':['gmst',0,'hadcrut5','ihadcrut5_global.dat'],
    'cpc_mjo3':['mjo',3,'ncep','icpc_mjo03_mean12.dat'],
    'cpc_mjo6':['mjo',6,'ncep','icpc_mjo06_mean12.dat'],
    }, index = ['name','subindex','product','climexpfile']) 

datapath = Path('/scistor/ivm/jsn295/Medi/monthly')

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
        with open(datapath / f'{name}.metadata', mode = 'wt') as mf:
            for line in rf:
                if line.startswith('#'):
                    mf.write(line)
                else:
                    content = line.splitlines()[0] # Can lead to an empty string if only \n
                    if content:
                        content = [float(s) for s in content.split(' ') if s] 
                        timestamps.append(content[0])
                        values.extend(content[1:])
    return timestamps, values

def decimal_year_to_datetime(decyear: float) -> datetime:
    """Decimal year to datetime, not accounting for leap years"""
    baseyear = int(decyear)
    ndays = 365 * (decyear - baseyear)
    return datetime(baseyear,1,1) + timedelta(days = ndays)

def process(name: str, timestamps: array.array, values: array.array) -> pd.DataFrame:
    """
    Missing value handling, and handling the case with multiple monthly values for one yearly timestamp
    """
    # Missing values
    values = np.array(values)
    values[np.isclose(values, np.full_like(values, -999.9))] = np.nan
    # Temporal index
    assert (np.allclose(np.diff(timestamps),1.0) or np.allclose(np.diff(timestamps), 1/12, atol = 0.001)), 'timestamps do not seem to be decimal years, with a yearly or monthly interval, check continuity'
    if len(timestamps) != len(values):
        assert (len(values) % len(timestamps)) == 0, 'values are not devisible by timestamps, check shapes and lengths'
        warnings.warn(f'Spotted one timestamp per {len(values)/len(timestamps)} values data')
    timestamps = pd.date_range(start = decimal_year_to_datetime(timestamps[0]), periods = len(values),freq = 'MS') # Left stamped
    series = pd.DataFrame(values[:,np.newaxis], index = timestamps)
    # Adding extra information, except for climexp file
    series.columns = pd.MultiIndex.from_tuples([tuple(climexp_properties[name].values)],names = climexp_properties.index)
    series.columns = series.columns.droplevel(-1)
    return series 

def main() -> pd.DataFrame:
    """Collecting all the ready timeseries data"""
    collection = []
    for name in climexp_properties.keys():
        print(f'starting with: {name}')
        t1, v1 = download_raw_data(name)
        collection.append(process(name, t1, v1))
    return pd.concat(collection, axis = 1, join = 'outer') 

complete = main()
#t1, v1 = download_raw_data('hadsst_pdo')
#s1 = process('hadsst_pdo',t1,v1)

# TODO: derive series from reanalysis fields. 
# - subtropical jet strength
# Download manually for now

