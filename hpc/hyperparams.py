import os
import sys
import json
import uuid
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
from Medley.preprocessing import Anomalizer, remove_bottleneck, make_pipeline
from Medley.dataloading import prep_and_resample
from Medley.estimators import return_estimator
from Medley.crossval import SpatiotemporalSplit
from Medley.utils import regions
from Medley.interpretation import load_pred_results

warnings.simplefilter('ignore',category=RuntimeWarning)
warnings.simplefilter('ignore',category=UserWarning)

n_jobs = int(sys.argv[1])

experimentpath = Path('/scistor/ivm/jsn295/Medi/hyperparams/')

experiment = dict( 
    region_name = 'medwest',
    prep_kwargs= dict(
        target_var = 'SPI1',
        minsamples = 10, # numer of stations
        resampling = 'multi', # whether multiple targets / samples are desired per anchor year
        shift = False, # 
        resampling_kwargs = dict(
            precursor_agg = 1, # Number of months
            n = 2, # number of lags
            separation = 0, #step per lag
            target_agg = 1, # ignored if resampling == 'single', as aggregation will be based on first/last, also questionable if useful with 3-month SPI
            firstmonth = 1, # How to define the winter period (with lastmonth)
            lastmonth = 3,
            ),
        ),
    predictor_kwargs = dict( # Fixed arguments
        npreds = 20,
        expid = 'fa83d256b8', #'5f53f2f833', # 
        subdir = None, #'pre0512_config', #
        ),
    #predictor_kwargs = dict(),
    bottleneck_kwargs = dict(
        startyear = 1950, # To remove bottleneck data
        endyear = 2023,
        fraction_valid = 0.8, # Fraction non-nan required in desired window
        ),
    cv_kwargs = dict(
        n_temporal=5,
        ),
    #estimator = 'rfreg',
    #estimator_kwargs = dict( # Fixed arguments
    #    n_estimators = 1500,
    #    ),
    estimator = 'xgbreg',
    estimator_kwargs = dict( # Fixed arguments
        n_jobs=1,
        ),
    #pipeline_kwargs = dict( # Further processing after the
    #    anom = False,
    #    scale = False,
    #    ),
    pipeline_kwargs = dict(),
    #studyspace = dict( # Tuple with upper and lower bounds of parameters, or lists of options
    #    min_samples_split= ((0.01,0.1),False),
    #    max_features= ((0.1,1.0),False),
    #    max_depth=[3,5,8,11],
    #    ),
    studyspace = dict( # Tuple with upper and lower bounds of parameters plus whether logarithmic, or lists of options
        n_estimators = ((50,500),False),
        learning_rate= ((0.001,0.1),True),
        max_depth=((1,10),False),
        ),
    studykwargs = dict(
        score='neg_mean_squared_error',
        direction='maximize',
        n_trials=200,
        n_jobs=n_jobs,
        ),
    )

def trial_objective(trial, X, y, cv, estimator: str, estimator_kwargs: dict, studyspace: dict, score: str = 'r2'):
    """
    Trial model is initialized here,
    fitted and evaluated in cv mode
    returns average score
    """
    hyperparams = deepcopy(estimator_kwargs) 
    for paramname, space in studyspace.items():
        if isinstance(space, tuple): # lower, upper bounds, and whether logarithmic
            (lower, upper), log = space 
            if isinstance(lower,int):
                hyperparams[paramname] = trial.suggest_int(paramname, lower, upper, log=log)
            else:
                hyperparams[paramname] = trial.suggest_float(paramname, lower, upper, log=log)
        elif isinstance(space,list):
            hyperparams[paramname] = trial.suggest_categorical(paramname, space)
    print(hyperparams)
    modelclass = return_estimator(estimator)
    model = modelclass(**hyperparams)
    
    scores = cross_val_score(model, X, y, scoring = score, cv=cv, n_jobs = 1)
    return scores.mean()


def main(region_name, prep_kwargs, predictor_kwargs, bottleneck_kwargs, cv_kwargs, estimator, estimator_kwargs, pipeline_kwargs, studyspace, studykwargs):
    """
    Data loading and resampling wrapped in preparation function
    """
    prep_kwargs.update({'target_region':regions[region_name]})
    Xm, ym, cal = prep_and_resample(**prep_kwargs)

    if predictor_kwargs:
        npreds = predictor_kwargs.pop('npreds')
        result, cv_scores = load_pred_results(**predictor_kwargs)
        prednames = result.loc[npreds,'feature_names']
        Xm = Xm.loc[:,list(prednames)]

    # Years from which data is required, dropping bottleneck variables
    # EKE only 1980-2018
    # MJO only 1980-now
    # AMOC only 2004-2020
    Xm, ym = remove_bottleneck(Xm, ym, **bottleneck_kwargs)

    # cv has to be an iterator, providing train and test indices. Can still overlap from season to season
    # Todo: make contigouus and seasonal?
    cv_kwargs['time_dim'] = Xm.index  
    cv = SpatiotemporalSplit(**cv_kwargs)
    if pipeline_kwargs: # Building in some preprocessing steps that should be crossvalidated
        model = make_pipeline(estimator = model, **pipeline_kwargs)
    study = optuna.create_study(direction=studykwargs.pop('direction')) # default is TPESampler
    score = studykwargs.pop('score')
    #return Xm, ym, score, cv
    study.optimize(partial(trial_objective, X=Xm, y=ym.squeeze(),cv=cv, estimator=estimator, estimator_kwargs=estimator_kwargs, studyspace=studyspace, score=score),**studykwargs) 
    return study, cv 

if __name__ == "__main__":
    expid = uuid.uuid4().hex[:10]
    print(experiment)
    with open(experimentpath / f'{expid}_experiment.json', mode = 'wt') as f:
        json.dump(experiment, fp = f)
    study, cv = main(**experiment)
    results = study.trials_dataframe()
    results.to_csv(experimentpath / f'{expid}_results.csv', sep = ';')
