import os
import sys
import json
import warnings
import optuna
import xarray as xr
import numpy as np
import pandas as pd

from pathlib import Path
from joblib import parallel_backend
from copy import deepcopy
from functools import partial

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

sys.path.append(os.path.expanduser('~/Documents/Medley'))
from Medley.preprocessing import Anomalizer
from Medley.dataloading import prep_and_resample
from Medley.crossval import SpatiotemporalSplit

warnings.simplefilter('ignore',category=RuntimeWarning)
warnings.simplefilter('ignore',category=UserWarning)

experimentpath = Path('/scistor/ivm/jsn295/Medi/hyperparams/')

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
    estimator_kwargs = dict( # Fixed arguments
        n_estimators = 500,
        ),
    studyspace = dict( # Tuple with upper and lower bounds of parameters, or lists of options
        min_samples_split= (2,6),
        max_features= (0.5,1.0),
        max_depth=[3,8],
        ),
    studykwargs = dict(
        score='r2',
        direction='maximize',
        n_trials=3,
        n_jobs=10,
        ),
    )

def trial_objective(trial, X, y, cv, estimator_kwargs: dict, studyspace: dict, score: str = 'r2'):
    """
    Trial model is initialized here,
    fitted and evaluated in cv mode
    returns average score
    """
    hyperparams = deepcopy(estimator_kwargs) 
    for paramname, space in studyspace.items():
        if isinstance(space, tuple): # lower, upper bounds
            lower, upper = space 
            if isinstance(lower,int):
                hyperparams[paramname] = trial.suggest_int(paramname, lower, upper, log=False)
            else:
                hyperparams[paramname] = trial.suggest_float(paramname, lower, upper, log=False)
        elif isinstance(space,list):
            hyperparams[paramname] = trial.suggest_categorical(paramname, space)
    print(hyperparams)
    model = RandomForestRegressor(**hyperparams)
    
    scores = cross_val_score(model, X, y, scoring = score, cv=cv, n_jobs = 1)
    return scores.mean()


def main(prep_kwargs, startyear, endyear, fraction_valid, cv_kwargs, estimator_kwargs, studyspace, studykwargs):
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

    # cv has to be an iterator, providing train and test indices. Can still overlap from season to season
    # Todo: make contigouus and seasonal?
    cv_kwargs['time_dim'] = Xm.index  
    cv = SpatiotemporalSplit(**cv_kwargs)
    study = optuna.create_study(direction=studykwargs.pop('direction'))
    score = studykwargs.pop('score')
    #return Xm, ym, score, cv
    study.optimize(partial(trial_objective, X=Xm, y=ym.squeeze(),cv=cv, estimator_kwargs=estimator_kwargs, studyspace=studyspace),**studykwargs) 
    return study, cv 

if __name__ == "__main__":
    print(experiment)
    with open(experimentpath / 'experiment.json', mode = 'wt') as f:
        json.dump(experiment, fp = f)
    study, cv = main(**experiment)
    #Xm, ym, score, cv = main(**experiment)
