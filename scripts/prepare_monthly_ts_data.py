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
    'stat_nao':['nao',0,'station','inao.dat'], # 12 times a monthly value, then an annual value. https://crudata.uea.ac.uk/cru/data/nao/
    'ncar_snao':['snao',0,'ncep','isnao_ucar.dat'],
    'cpc_ao':['ao',0,'ncep','icpc_ao.dat'], # HAs a two-part time index
    'slp_ao':['ao',0,'trenb','iao_slp_ext.dat'],
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
                        if name == 'cpc_ao':
                            timestamps.append(content.pop(0) + (content.pop(0)-1)/12) # year and a month to decimal year
                        else:
                            timestamps.append(content.pop(0))
                        if name == 'stat_nao':
                            content.pop(-1) # Last one is an annual value
                        values.extend(content)
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

def download_climexp() -> pd.DataFrame:
    """Collecting all the ready timeseries data from climexp"""
    collection = []
    for name in climexp_properties.keys():
        print(f'starting with: {name}')
        t1, v1 = download_raw_data(name)
        ts = process(name, t1, v1)
        collection.append(ts)
    return pd.concat(collection, axis = 1, join = 'outer') 

def u250_series() -> pd.DataFrame:
    """
    Monthly mean zonal mean u250
    Extracting latitude of maximum, and strengths at
    40N and 55N
    """
    da = xr.open_dataarray(datapath / f'monthly_zonalmean_u250_NH_-25E_50E.nc')
    collection = da.idxmax('latitude').to_dataframe()
    collection.columns = pd.MultiIndex.from_tuples([('lat_u250max',0,'era5')], names = climexp_properties.index[:-1])
    for lat in [30,40,55]:
        ts = da.sel(latitude = lat, drop = True).to_dataframe().astype(np.float32)
        ts.columns = pd.MultiIndex.from_tuples([('u250',lat,'era5')], names = climexp_properties.index[:-1])
        collection = collection.join(ts, how = 'left') # Indices should match so how is irrelevant.
    return collection

def get_monthly_data(force_update: bool = False):
    """
    Main function, reads data if stored
    else will run the downloading and processing (or when update is forced).
    """
    finalpath = datapath / 'complete.parquet'
    if (not finalpath.exists()) or force_update:
        climexp_df = download_climexp()
        era_df = u250_series()
        complete = climexp_df.join(era_df, how = 'outer') # Should not add rows to climexp_df as that is much larger.
        table = pa.Table.from_pandas(complete) # integer multiindex level becomes text
        pq.write_table(table, finalpath)
    else:
        table = pq.read_table(finalpath)
    return table
