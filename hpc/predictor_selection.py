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
from Medley.preprocessing import Anomalizer, remove_bottleneck, make_pipeline
from Medley.dataloading import prep_and_resample
from Medley.estimators import return_estimator
from Medley.crossval import SpatiotemporalSplit
from Medley.utils import regions

warnings.simplefilter('ignore',category=RuntimeWarning)
warnings.simplefilter('ignore',category=UserWarning)

n_jobs = int(sys.argv[1])

datapath = Path('/scistor/ivm/jsn295/Medi/monthly/')
experimentpath = Path('/scistor/ivm/jsn295/Medi/predselec/')

experiment = dict( 
    region_name = 'medwest',
    prep_kwargs= dict(
        target_var = 'SPI3',
        minsamples = 10, # numer of stations
        resampling = 'multi', # whether multiple targets / samples are desired per anchor year
        shift = False, # 
        resampling_kwargs = dict(
            precursor_agg = 1, # Number of months
            n = 2, # number of lags
            separation = 0, #step per lag
            target_agg = 1, # ignored if resampling == 'single', as aggregation will be based on first/last, also questionable if useful with 3-month SPI
            firstmonth = 12, # How to define the winter period (with lastmonth)
            lastmonth = 3,
            ),
        ),
    bottleneck_kwargs = dict(
        startyear = 1950, # To remove bottleneck data
        endyear = 2023,
        fraction_valid = 0.8, # Fraction non-nan required in desired window
        ),
    cv_kwargs = dict(
        n_temporal=5,
        ),
    #estimator = 'ridreg',
    #estimator_kwargs = dict(),
    estimator = 'rfreg',
    estimator_kwargs = dict(
        n_estimators = 1500,
        max_depth = 6,
    #    #learning_rate = 0.01,
    #    #n_jobs=n_jobs,
        min_samples_split=0.01, # With max about 200 samples, anything below 0.01 does not make sense
        max_features = 0.3,
        ),
    #pipeline_kwargs = dict( # Further processing after the
    #    anom = False,
    #    scale = False,
    #    ),
    pipeline_kwargs = dict(),
    sequential_kwargs = dict(
        k_features=1,
        forward=False,
        scoring='neg_mean_squared_error',
        n_jobs=n_jobs,
        ),
    )

class DataFrameWrapper(pd.DataFrame):
    """
    Workaround class for the very inconvenient fact
    that Anomalizer needs access to the Index.
    https://github.com/rasbt/mlxtend/issues/943
    """
    @property
    def values(self):
        return self

    def __getitem__(self, indexers):
        if isinstance(indexers, tuple):
            indexers = tuple([
                idx if not isinstance(idx, tuple) else list(idx)
                for idx in indexers
            ])
        return self.iloc[indexers]

    def copy(self, deep = True):
        """
        Needed because otherwise copy returns a newly constructed dataframe
        And copy is called here https://github.com/rasbt/mlxtend/blob/77a9a27ffd9c70e6099859828e678a1988420b8c/mlxtend/feature_selection/utilities.py#L115
        """
        return DataFrameWrapper(super().copy(deep = deep))


def main(region_name, prep_kwargs, bottleneck_kwargs, cv_kwargs, estimator, estimator_kwargs, pipeline_kwargs, sequential_kwargs):
    """
    Data loading and resampling wrapped in preparation function
    """
    prep_kwargs.update({'target_region':regions[region_name]})
    Xm, ym, cal = prep_and_resample(**prep_kwargs)

    # Years from which data is required, dropping bottleneck variables
    # EKE only 1980-2018
    # MJO only 1980-now
    # AMOC only 2004-2020
    Xm, ym = remove_bottleneck(Xm, ym, **bottleneck_kwargs)
    modelclass = return_estimator(estimator)
    model = modelclass(**estimator_kwargs)
    # cv has to be an iterator, providing train and test indices. Can still overlap from season to season
    # Todo: make contigouus and seasonal?
    cv_kwargs['time_dim'] = Xm.index  
    cv = SpatiotemporalSplit(**cv_kwargs)
    if pipeline_kwargs: # Building in some preprocessing steps that should be crossvalidated
        model = make_pipeline(estimator = model, **pipeline_kwargs)
        if 'anom' in model.named_steps: # Workaround needed to pass DataFrame
            Xm = DataFrameWrapper(Xm)
    if estimator.endswith('resreg') and (not isinstance(Xm,DataFrameWrapper)):
         Xm = DataFrameWrapper(Xm)
    sfs = SequentialFeatureSelector(estimator = model, cv = cv, **sequential_kwargs)
    sfs.fit(X = Xm, y=ym.squeeze())
    return sfs, cv, Xm

if __name__ == "__main__":
    expid = uuid.uuid4().hex[:10]
    print(experiment)
    with open(experimentpath / f'{expid}_experiment.json', mode = 'wt') as f:
        json.dump(experiment, fp = f)
    sfs, cv, Xm = main(**experiment)
    results = pd.DataFrame(sfs.get_metric_dict()).T
    results.to_csv(experimentpath / f'{expid}_results.csv', sep = ';')
