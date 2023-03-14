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
    'ersst_nino_34':['enso',3,'iersst_nino3.4a_rel.dat'],
    'ersst_nino_4':['enso',4,'iersst_nino4a_rel.dat'],
    'ersst_amo':['amo',0,'iamo_ersst.dat'],
    }, index = ['name','subindex','climexpfile']) 

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

def process(name: str, timestamps: array.array, values: array.array) -> pd.Series:
    """
    Missing value handling, and handling the case with multiple monthly values for one yearly timestamp
    """
    values = np.array(values)
    values[np.isclose(values, np.full_like(values, -999.9))] = np.nan
    assert (np.allclose(np.diff(timestamps),1.0) or np.allclose(np.diff(timestamps), 1/12, atol = 0.001)), 'timestamps do not seem to be decimal years, with a yearly or monthly interval, check continuity'
    if len(timestamps) != len(values):
        assert (len(values) % len(timestamps)) == 0, 'values are not devisible by timestamps, check shapes and lengths'
        warnings.warn(f'Spotted one timestamp per {len(values)/len(timestamps)} values data')
    timestamps = pd.date_range(start = decimal_year_to_datetime(timestamps[0]), periods = len(values),freq = 'M')
    return pd.Series(values, index = timestamps)

t1, v1 = download_raw_data('ersst_nino_4')
t2, v2 = download_raw_data('ersst_amo')

s1 = process('a', t1, v1)
s2 = process('b', t2, v2)

# Todo, adding info and keys in process function.
