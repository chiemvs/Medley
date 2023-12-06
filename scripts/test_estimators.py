import os
import sys
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.append(os.path.expanduser('~/Documents/Medley'))

from Medley.preprocessing import Anomalizer, remove_bottleneck, make_pipeline
from Medley.dataloading import prep_and_resample, prep_ecad, get_monthly_data
from Medley.estimators import return_estimator
from Medley.crossval import SpatiotemporalSplit
from Medley.interpretation import load_pred_results, return_explainer
from Medley.utils import regions

warnings.simplefilter('ignore',category=RuntimeWarning)
warnings.simplefilter('ignore',category=UserWarning)

prep_kwargs= dict(
    target_region = regions['medwest'],
    target_var = 'SPI1',
    minsamples = 10, # numer of stations
    resampling = 'multi', # whether multiple targets / samples are desired per anchor year
    shift = False,
    resampling_kwargs = dict(
        precursor_agg = 1, # Number of months
        n = 2, # number of lags
        separation = 0, #step per lag
        target_agg = 1, # ignored if resampling == 'single', as aggregation will be based on first/last
        firstmonth = 1, # How to define the winter period (with lastmonth)
        lastmonth = 3,
        ),
    )
predictor_kwargs = dict( # Fixed arguments
    npreds = 7,
    expid = 'dd0d579f41', #'fa83d256b8', #'5f53f2f833', # 
    subdir = None, #'pre0512_config', #
    )
bottleneck_kwargs = dict(
    startyear = 1980, # To remove bottleneck data
    endyear = 2023,
    fraction_valid = 0.8, # Fraction non-nan required in desired window
    )
cv_kwargs = dict(
    n_temporal=5,
    )
#estimator = 'xgbreg'
#estimator_kwargs = dict(
#    n_estimators = 345,
#    learning_rate= 0.09,
#    max_depth = 3,
#    n_jobs = 20,
#    )
estimator = 'rfreg'
estimator_kwargs = dict(
    n_estimators = 1500,
    max_depth = 11,
    min_samples_split=0.04, # With max about 200 samples, anything below 0.01 does not make sense
    max_features = 1.0,
    )
pipeline_kwargs = dict(
    anom = False,
    scale = False,
    )

if __name__ == '__main__':
    X, y, cal = prep_and_resample(**prep_kwargs)
    #y = prep_ecad(prep_kwargs['target_region'], 'SPI1', shift = False).to_frame()
    #X = get_monthly_data()

    ## Extraction of selected predictors, from a good experiment
    if predictor_kwargs:
        npreds = predictor_kwargs.pop('npreds')
        result, cv_scores = load_pred_results(**predictor_kwargs)
        prednames = result.loc[npreds,'feature_names']
        X = X.loc[:,list(prednames)]

    X, y = remove_bottleneck(X, y, **bottleneck_kwargs)
    modelclass = return_estimator(estimator)
    model = modelclass(**estimator_kwargs)
    #model.fit(X,y.squeeze())
    #yhat = model.predict(X)

    #expl = return_explainer(model)
    #attribution = expl.explain(X)

    cv_kwargs['time_dim'] = X.index  
    cv = SpatiotemporalSplit(**cv_kwargs)

    #p = make_pipeline(estimator = model, **pipeline_kwargs)
    #p.fit(X, y.squeeze()) # Some conversion to numpy takes place
    #expl2 = return_explainer(p)
    #attribution2 = expl2.explain(X)


    #yhats = cross_val_predict(p, X = X, y = y.squeeze(), cv = cv)
    yhats2 = cross_val_predict(model, X = X, y = y.squeeze(), cv = cv)
    print(r2_score(y,yhats2))
    #scores = cross_val_score(model, X = X, y = y.squeeze(), cv = cv, scoring = 'neg_mean_absolute_error')
    scores = cross_validate(model, X = X, y = y.squeeze(), cv = cv, scoring = ['r2','neg_mean_squared_error'])
    print(scores['test_neg_mean_squared_error'].mean())
    #scores2 = cross_validate(p, X = X, y = y.squeeze(), cv = cv, scoring = ['r2','neg_mean_absolute_error'])

