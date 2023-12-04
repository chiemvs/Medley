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
ECAD manual recomputation code
"""
testdf = pd.read_hdf(datapath / f'eca_preaggregated_RR.h5').dropna(axis = 1, how = 'all')
testdf.index.name = 'time'
testts = testdf[21] 

#spits = transform_to_spi(testts, climate_period = slice('1990-01-01','2000-01-01'))
#spits = transform_to_spi(testts)
spi = testdf.apply(transform_to_spi, axis = 0)

spi.to_hdf(datapath / 'eca_preaggregated_SPI1.h5', key = 'SPI1', mode = 'w')
#os.system(f'cp {datapath / "eca_preaggregated_SPI3_stations.h5"} {datapath / "eca_preaggregated_SPI1_stations.h5"}')

