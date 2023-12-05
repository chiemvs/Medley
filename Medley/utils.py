import warnings
import array
import numpy as np
import xarray as xr
import pandas as pd

from datetime import datetime, timedelta
from typing import Union
from scipy import stats

# Domain splitting for u to capture subtropical jet and eddy-driven jet separately.
udomains = {'med':(-8.5,42), # Portugal to eastern turkey
        'atl':(-50,-10)} # from Newfoundland coast to Ireland coast

west = dict(include = {
    'iberia':(-9.8,35.98,3.6,43.8),
    'france_italy':(4,43,13.4,45.63),
    },
    exclude = {
    'islands':(0.79,35.2,4.6,40.4),
    'north_africa':(-1.450,34.457,11.217,36.972),
    })

medwest = dict(include = {
    'iberia':(-9.8,35.98,8,44.1),
    'italy':(8,35,18.6,46),
    },
    exclude = {
    'north_africa':(-1.450,34.457,11.217,36.972),
    'eastadriatic':(15,42.2,20,48),
    'slovenia':(14,44.1,20,48),
    'tunesia':(5,30,12,38),
    'pyrenees':(-2,41.8,2.4,45.63),
    #'alps':(7.5,44.9,11,45.63),
    })

centraleast = dict(include = {
    'greeceplus':(19,34,28.5,42.7),
    },
    exclude = {})

east = dict(include = {
    'turkeycyprus':(29,32,37,38),
    },
    exclude = {})

medeast = dict(include = {
    'greeceplus':(18.9,34,37,42.7),
    'israel':(32,30,37,38),
    },
    exclude = {})

regions = {'west':west,'medwest':medwest,'centraleast':centraleast,'east':east,'medeast':medeast}

tscolnames = ['name','subindex','product']

def fit_gamma(rr : np.ndarray, minsamples: int = 20) -> tuple[float,float,float]:
    """
    Returns a fitted 3-parameter gamma distribution
    initial guess for loc parameter is zero
    Returns False when fitting fails or when too little samples
    """
    if (len(rr) >= minsamples):
        try:
            gamparams = stats.gamma.fit(rr, loc=0)
        except stats._warnings_errors.FitError:
            warnings.warn('fitting gamma failed')
            gamparams = False
    else:
        warnings.warn('detected lower than minimum amount of samples')
        gamparams = False
    return gamparams


def spi_from_monthly_rainfall(rr : Union[np.ndarray,xr.DataArray,pd.Series], minsamples: int = 20, gamparams: tuple = None) -> Union[np.ndarray,pd.Series]:
    """
    Should be monthly accumulation, normalization with respect to entire data
    so make sure you supply only values for same-month-in-year
    Stagge, J. H., Tallaksen, L. M., Gudmundsson, L., Van Loon, A. F., & Stahl, K. (2015). Candidate distributions for climatological drought indices (SPI and SPEI). International Journal of Climatology, 35(13), 4027-4040.
    Returns NA for too little samples
    Possible to supply pre-computed gamma parameters. (False if presupplied fit failed)
    """
    assert (not isinstance(rr, pd.DataFrame)), 'Only 1-D data should be supplied'
    if isinstance(rr, pd.Series):
        restore = pd.Series(np.nan, index = rr.index, name = 'SPI')
        rr = rr.values
    elif isinstance(rr, xr.DataArray):
        assert rr.ndim == 1, 'Only 1-D xarray should be supplied'
        restore = xr.DataArray(np.nan, coords = rr.coords, dims = rr.dims, name = 'SPI')
        rr = rr.values
    else:
        restore = False
    if gamparams is None:
        gamparams = fit_gamma(rr, minsamples = minsamples) # attempt fit parameters of gamma distribution to SPI data, will become False if too little samples or unsuccesful
    if gamparams:
        rv = stats.gamma(*gamparams) # Continuous random variable class, can sample randomly from the gamma distribution we just fitted
        # Account for zero values (cfr.Stagge et al. 2015))
        indices_nonzero = np.nonzero(rr)[0]
        nsamples_zero = len(rr) - np.count_nonzero(rr)
        ratio_zeros = nsamples_zero / len(rr)

        p_zero_mean = (nsamples_zero + 1) / (2 * (len(rr) + 1))
        # Creating a series equal in length to the 1-month sums, which will receive the probabilities
        prob_gamma = np.full_like(a = rr, fill_value = p_zero_mean)
        # overwriting the p_zero for all non-zero months
        prob_gamma[indices_nonzero] = ratio_zeros+((1-ratio_zeros)*rv.cdf(rr[indices_nonzero]))

        # Step 3:
        # Transform Gamma probabilities to standard normal probabilities (plus clipping deviations greater than 3 std)
        z_std = stats.norm.ppf(prob_gamma)
        z_std[z_std>3] = 3
        z_std[z_std<-3] = -3
    else:
        warnings.warn('gamparams was False, filling with NA')
        z_std = np.full_like(a = rr, fill_value = np.nan)
    if isinstance(restore, (pd.Series,xr.DataArray)):
        restore[:] = z_std
        return restore
    else:
        return z_std

def transform_to_spi(rr : Union[pd.Series,xr.DataArray], minsamples: int = 20, climate_period : slice = None):
    """
    Handles a single timeseries, removing na's.
    Possible to harmonize the climate period
    Might be handy for station data with different periods
    applies the transformation per month
    """
    def per_month(rr, climate_period, minsamples, stack: bool = False):
        spikwargs = dict(minsamples = minsamples)
        if not (climate_period is None):
            assert not stack, 'members in combination with climate period is not supported'
            climdat = rr.loc[climate_period] 
            spikwargs.update({'gamparams': fit_gamma(climdat.values, minsamples = minsamples)})
        else:
            spikwargs.update({'gamparams': None})
        if stack:
            stackdims = [d for d in ['amoc','member','time'] if d in rr.dims]
            result = spi_from_monthly_rainfall(rr = rr.stack({'stacked':stackdims}), **spikwargs) 
            return result.unstack('stacked')
        else:
            return spi_from_monthly_rainfall(rr = rr, **spikwargs) 

    if isinstance(rr, xr.DataArray):
        rr = rr.dropna(dim = 'time')
        spi = rr.groupby(rr.time.dt.month).apply(per_month, climate_period = climate_period, minsamples = minsamples, stack = 'member' in rr.dims)
    else:
        rr = rr.dropna()
        spi = rr.groupby(rr.index.month).apply(per_month, climate_period = climate_period, minsamples = minsamples)
        spi.index = spi.index.droplevel(0)
        spi = spi.sort_index()
    return spi


def chunk_func(ds: xr.Dataset, chunks = {'latitude':50, 'longitude':50}) -> xr.Dataset:
    """
    Chunking only the spatial dimensions. Eases reading complete timeseries at one grid location. 
    """
    #ds = ds.transpose("time","latitude", "longitude")
    chunks.update({'time':len(ds.time)}) # Needs to be specified as well, otherwise chunk of size 1.
    return ds.chunk(chunks)

def decimal_year_to_datetime(decyear: float) -> datetime:
    """Decimal year to datetime, not accounting for leap years"""
    baseyear = int(decyear)
    ndays = 365 * (decyear - baseyear)
    return datetime(baseyear,1,1) + timedelta(days = ndays)

def process_ascii(timestamps: array.array, values: array.array, miss_val = -999.9) -> pd.DataFrame:
    """
    both timestamps and values should be 1D
    Missing value handling, and handling the case with multiple monthly values for one yearly timestamp
    """
    # Missing values
    values = np.array(values)
    ismiss = np.isclose(values, np.full_like(values, miss_val))
    values = values.astype(np.float32) # Conversion to float because of np.nan
    values[ismiss] = np.nan
    # Temporal index
    assert (np.allclose(np.diff(timestamps),1.0) or np.allclose(np.diff(timestamps), 1/12, atol = 0.001)), 'timestamps do not seem to be decimal years, with a yearly or monthly interval, check continuity'
    if len(timestamps) != len(values):
        assert (len(values) % len(timestamps)) == 0, 'values are not devisible by timestamps, check shapes and lengths'
        warnings.warn(f'Spotted one timestamp per {len(values)/len(timestamps)} values data')
    timestamps = pd.date_range(start = decimal_year_to_datetime(timestamps[0]), periods = len(values),freq = 'MS') # Left stamped
    series = pd.DataFrame(values[:,np.newaxis], index = timestamps)
    # Adding extra information, except for climexp file
    return series 

def coord_to_decimal_coord(coord: str):
    """
    e.g. +015:58:41 to decimal coords
    but also +45:49:00
    and -000:41:29
    """
    assert coord[0] in ['+','-']
    degree = int(coord[1:-6]) 
    minutes = int(coord[-5:-3])
    seconds = int(coord[-2:])
    decimal = abs(degree) + minutes/60 + seconds/3600
    if coord[0] == '+':
        return decimal
    else:
        return -decimal
