import sys
import os
import xarray as xr
import numpy as np
import pandas as pd
import pyarrow as pa

from pathlib import Path
from typing import Union
from scipy import stats

sys.path.append(os.path.expanduser('~/Documents/Medley/'))
from Medley.utils import chunk_func, spi_from_monthly_rainfall, fit_gamma, transform_to_spi, regions
from Medley.preprocessing import monthly_resample_func, makemask
from Medley.dataloading import datapath

"""
SPI1 is not standard provided in the ECA&D project
therefore this is my own code to calculate SPI1 from monthly rainfall accumulations
This is in addition to SPI1 data provided by Gerard from KNMI (SPI1_Gerard) 
"""
testdf = pd.read_hdf(datapath / f'eca_preaggregated_RR.h5').dropna(axis = 1, how = 'all')
testdf.index.name = 'time'

#spi = testdf.apply(transform_to_spi, axis = 0)
#spi.to_hdf(datapath / 'eca_preaggregated_SPI1.h5', key = 'SPI1', mode = 'w')

#spi = pd.read_hdf(datapath / 'eca_preaggregated_SPI1.h5', key = 'SPI1')

#old_locs = pd.read_hdf(f'{datapath / "eca_preaggregated_SPI3_stations.h5"}')
#remaining_locs = old_locs.loc[spi.columns,:]
#remaining_locs.to_hdf(datapath / "eca_preaggregated_SPI1_stations.h5", mode = 'w', key = 'SPI1')

