import os
import sys
import json
import warnings
import xarray as xr
import numpy as np
import pandas as pd

from pathlib import Path
from joblib import parallel_backend

from sklearn.ensemble import RandomForestRegressor
from mlxtend.feature_selection import SequentialFeatureSelector

sys.path.append(os.path.expanduser('~/Documents/Medley'))
from scripts.prepare_monthly_ts_data import get_monthly_data
from Medley.utils import tscolnames
from Medley.preprocessing import Anomalizer, single_target_lagged_resample, simultaneous_resample, makemask, average_within_mask, multi_target_lagged_resample

warnings.simplefilter('ignore',category=RuntimeWarning)
warnings.simplefilter('ignore',category=UserWarning)

datapath = Path('/scistor/ivm/jsn295/Medi/monthly/')
experimentpath = Path('/scistor/ivm/jsn295/Medi/predselec/')

experiment = dict( 
    target_region = dict(
        include = {'iberia':(-9.8,35.98,3.6,43.8), 'france_italy':(4,43,13.4,45.63),},
        exclude = {'islands':(0.79,35.2,4.6,40.4),'north_africa':(-1.450,34.457,11.217,36.972),}
        ),
    target_var = 'RR',
    minsamples = 10, # numer of stations
    resampling = 'multi', # whether multiple targets / samples are desired per anchor year
    resampling_kwargs = dict(
        precursor_agg = 1, # Number of months
        n = 3, # number of lags
        separation = 0, #step per lag
        target_agg = 1, # ignored if resampling == 'multi'
        firstmonth = 12, # How to define the winter period (with lastmonth)
        lastmonth = 3,
        ),
    startyear = 1980, # To remove bottleneck data
    endyear = 2023,
    fraction_valid = 0.8, # Fraction non-nan required in desired window
    cv_kwargs = dict(
        a=None,
        ),
    estimator_kwargs = dict(
        ntrees = 500,
        ),
    sequential_kwargs = dict(
        k_features='parsimonious',
        forward=True,
        ),
    )


def main(target_region, target_var, minsamples, resampling, resampling_kwargs, startyear, endyear, fraction_valid, cv_kwargs, estimator_kwargs, sequential_kwargs):
    mask = makemask(target_region) 
    df = get_monthly_data(force_update = False).to_pandas()
    ecad = pd.read_hdf(datapath / f'eca_preaggregated_{target_var}.h5')
    ecad.index.name = 'time'
    ecad_locs = pd.read_hdf(datapath / f'eca_preaggregated_{target_var}_stations.h5')

    target = average_within_mask(mask = mask, data = ecad, datalocs = ecad_locs, minsamples=minsamples).to_frame()
    target.columns = pd.MultiIndex.from_tuples([(target_var,0,'ECAD')], names = tscolnames)

    # Define temporal sampling approaches
    if resampling == 'single':
        resampling_kwargs.pop('target_agg')
        Xm, ym, cm = single_target_lagged_resample(X = df, y = target, **resampling_kwargs) 
    elif resampling == 'multi':
        Xm, ym, cm = multi_target_lagged_resample(X = df, y = target, **resampling_kwargs) 
    else:
        raise ValueError('invalid resampling instruction given, should be one of "multi" or "single"')

    # Years from which data is required, dropping bottleneck variables
    # EKE only 1980-2018
    # MJO only 1980-now
    # AMOC only 2004-2020
    timeslice = slice(startyear,endyear)
    Xm = Xm.loc[timeslice,:]
    insufficient = Xm.columns[Xm.count(axis = 0) < (len(Xm)*fraction_valid)]
    print(f'dropping predictors: {insufficient}')
    Xm = Xm.drop(insufficient, axis = 1).dropna()
    ym = ym.loc[Xm.index,:]

    model = RandomForestRegressor()
    
    # cv has to be an iterator
    #sfs = SequentialFeatureSelector(model = model, cv = cv, **sequential_kwargs)
    return Xm, ym

if __name__ == "__main__":
    print(experiment)
    with open(experimentpath / 'experiment.json', mode = 'wt') as f:
        json.dump(experiment, fp = f)
    Xm, ym = main(**experiment)
