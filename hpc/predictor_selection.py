import os
import sys
import json
import uuid
import warnings
import xarray as xr
import numpy as np
import pandas as pd

from pathlib import Path
from joblib import parallel_backend

from mlxtend.feature_selection import SequentialFeatureSelector

sys.path.append(os.path.expanduser('~/Documents/Medley'))
from Medley.preprocessing import Anomalizer, remove_X_bottleneck
from Medley.dataloading import prep_and_resample
from Medley.estimators import return_estimator
from Medley.crossval import SpatiotemporalSplit

warnings.simplefilter('ignore',category=RuntimeWarning)
warnings.simplefilter('ignore',category=UserWarning)

n_jobs = int(sys.argv[1])

datapath = Path('/scistor/ivm/jsn295/Medi/monthly/')
experimentpath = Path('/scistor/ivm/jsn295/Medi/predselec/')

experiment = dict( 
    prep_kwargs= dict(
        target_region = dict(
            include = {
                'iberia':(-9.8,35.98,8,44.6),
                'italy':(8,35,18,45.63),},
            exclude = {
                'north_africa':(-1.450,34.457,11.217,36.972),
                'eastadriatic':(15,43,20,48),
                'tunesia':(5,30,12,38),
                'pyrenees':(-2,41.8,3.7,45.63),}
            ),
        target_var = 'SPI3',
        minsamples = 10, # numer of stations
        resampling = 'multi', # whether multiple targets / samples are desired per anchor year
        resampling_kwargs = dict(
            precursor_agg = 1, # Number of months
            n = 1, # number of lags
            separation = 0, #step per lag
            target_agg = 1, # ignored if resampling == 'single', as aggregation will be based on first/last
            firstmonth = 1, # How to define the winter period (with lastmonth)
            lastmonth = 3,
            ),
        ),
    bottleneck_kwargs = dict(
        startyear = 1979, # To remove bottleneck data
        endyear = 2023,
        fraction_valid = 0.8, # Fraction non-nan required in desired window
        ),
    cv_kwargs = dict(
        n_temporal=5,
        ),
    estimator = 'linreg',
    estimator_kwargs = dict(),
    #estimator_kwargs = dict(
    #    n_estimators = 500,
    #    max_depth = 10,
    #    min_samples_split=0.01, # With max about 200 samples, anything below 0.01 does not make sense
    #    ),
    sequential_kwargs = dict(
        k_features=20,
        forward=True,
        scoring='r2',
        n_jobs=n_jobs,
        ),
    )


def main(prep_kwargs, bottleneck_kwargs, cv_kwargs, estimator, estimator_kwargs, sequential_kwargs):
    """
    Data loading and resampling wrapped in preparation function
    """
    Xm, ym, cal = prep_and_resample(**prep_kwargs)

    # Years from which data is required, dropping bottleneck variables
    # EKE only 1980-2018
    # MJO only 1980-now
    # AMOC only 2004-2020
    Xm, ym = remove_X_bottleneck(Xm, ym, **bottleneck_kwargs)
    modelclass = return_estimator(estimator)
    model = modelclass(**estimator_kwargs)
    # cv has to be an iterator, providing train and test indices. Can still overlap from season to season
    # Todo: make contigouus and seasonal?
    cv_kwargs['time_dim'] = Xm.index  
    cv = SpatiotemporalSplit(**cv_kwargs)
    sfs = SequentialFeatureSelector(estimator = model, cv = cv, **sequential_kwargs)
    sfs.fit(X = Xm, y=ym.squeeze())
    return sfs, cv 

if __name__ == "__main__":
    expid = uuid.uuid4().hex[:10]
    print(experiment)
    with open(experimentpath / f'{expid}_experiment.json', mode = 'wt') as f:
        json.dump(experiment, fp = f)
    sfs, cv = main(**experiment)
    results = pd.DataFrame(sfs.get_metric_dict()).T
    results.to_csv(experimentpath / f'{expid}_results.csv', sep = ';')
