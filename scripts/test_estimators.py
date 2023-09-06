import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_predict

sys.path.append(os.path.expanduser('~/Documents/Medley'))

from Medley.preprocessing import Anomalizer, remove_X_bottleneck
from Medley.dataloading import prep_and_resample, prep_ecad, get_monthly_data
from Medley.estimators import return_estimator
from Medley.crossval import SpatiotemporalSplit

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
    )
bottleneck_kwargs = dict(
    startyear = 1950, # To remove bottleneck data
    endyear = 2023,
    fraction_valid = 0.8, # Fraction non-nan required in desired window
    )
cv_kwargs = dict(
    n_temporal=5,
    )
estimator = 'linreg'
#estimator_kwargs = dict(
#    n_estimators = 500,
#    max_depth = 10,
#    min_samples_split=0.01, # With max about 200 samples, anything below 0.01 does not make sense
#    )
estimator_kwargs = dict()

if __name__ == '__main__':
    #X, y, cal = prep_and_resample(**prep_kwargs)
    y = prep_ecad(prep_kwargs['target_region'], 'SPI3').to_frame()
    X = get_monthly_data()

    X, y = remove_X_bottleneck(X, y, **bottleneck_kwargs)
    modelclass = return_estimator(estimator)
    model = modelclass(**estimator_kwargs)
    model.fit(X,y.squeeze())
    yhat = model.predict(X)

    #cv_kwargs['time_dim'] = X.index  
    #cv = SpatiotemporalSplit(**cv_kwargs)

    #yhats = cross_val_predict(model, X = X, y = y.squeeze(), cv = cv)
