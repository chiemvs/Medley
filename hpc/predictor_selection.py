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
from Medley.preprocessing import Anomalizer
from Medley.dataloading import prep_and_resample
from Medley.crossval import SpatiotemporalSplit

warnings.simplefilter('ignore',category=RuntimeWarning)
warnings.simplefilter('ignore',category=UserWarning)

datapath = Path('/scistor/ivm/jsn295/Medi/monthly/')
experimentpath = Path('/scistor/ivm/jsn295/Medi/predselec/')

experiment = dict( 
    prep_kwargs= dict(
        target_region = dict(
            include = {'iberia':(-9.8,35.98,3.6,43.8), 'france_italy':(4,43,13.4,45.63),},
            exclude = {'islands':(0.79,35.2,4.6,40.4),'north_africa':(-1.450,34.457,11.217,36.972),}
            ),
        target_var = 'RR',
        minsamples = 10, # numer of stations
        resampling = 'multi', # whether multiple targets / samples are desired per anchor year
        resampling_kwargs = dict(
            precursor_agg = 1, # Number of months
            n = 1, # number of lags
            separation = 0, #step per lag
            target_agg = 1, # ignored if resampling == 'multi'
            firstmonth = 11, # How to define the winter period (with lastmonth)
            lastmonth = 3,
            ),
        ),
    startyear = 1980, # To remove bottleneck data
    endyear = 2023,
    fraction_valid = 0.8, # Fraction non-nan required in desired window
    cv_kwargs = dict(
        n_temporal=10,
        ),
    estimator_kwargs = dict(
        n_estimators = 500,
        max_depth = 5,
        ),
    sequential_kwargs = dict(
        k_features=6,
        forward=True,
        scoring='r2',
        n_jobs=10,
        ),
    )


def main(prep_kwargs, startyear, endyear, fraction_valid, cv_kwargs, estimator_kwargs, sequential_kwargs):
    """
    Data loading and resampling wrapped in preparation function
    """
    Xm, ym, cal = prep_and_resample(**prep_kwargs)

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
    print(f'samples left: {ym.size}')
    print(f'features left: {Xm.shape[1]}')

    model = RandomForestRegressor(**estimator_kwargs)
    # cv has to be an iterator, providing train and test indices. Can still overlap from season to season
    # Todo: make contigouus and seasonal?
    cv_kwargs['time_dim'] = Xm.index  
    cv = SpatiotemporalSplit(**cv_kwargs)
    sfs = SequentialFeatureSelector(estimator = model, cv = cv, **sequential_kwargs)
    sfs.fit(X = Xm, y=ym.squeeze())
    return sfs, cv 

if __name__ == "__main__":
    print(experiment)
    with open(experimentpath / 'experiment.json', mode = 'wt') as f:
        json.dump(experiment, fp = f)
    sfs, cv = main(**experiment)
