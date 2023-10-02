import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import pyarrow as pa

from pathlib import Path
from typing import Union
from sklearn.pipeline import Pipeline

try:
    import shap
except ImportError:
    pass

from .estimators import _estimators

predselpath = Path('/scistor/ivm/jsn295/Medi/predselec/')
all_expids = [ p.name.split('_')[0] for p in predselpath.glob('*.json')]

def _extract_properties(expid : str) -> dict:
    with open(predselpath / f'{expid}_experiment.json',mode = 'rt') as f:
        dc = json.load(f)
    return dc

def _finditem(obj, key): 
    """https://stackoverflow.com/questions/14962485/finding-a-key-recursively-in-a-dictionary"""
    if key in obj: 
        return obj[key]
    for k, v in obj.items():
        if isinstance(v,dict):
            item = _finditem(v, key)
            if item is not None:
                return item

def find_experiments_with(key: str = None, value: Union[str, int] = None, keyval_dict = dict()) -> set:
    """
    Loops through experiments and returns those ids
    for which a value under the key equals value 
    at any level in the potentially nested hierarchy
    Multiple requirements can be given as a dictionary 
    """
    returnids = set(all_expids)
    if not keyval_dict: # Empty evaluates as False
        keyval_dict = {key:value}
    for key, value in keyval_dict.items():
        currentids = set()
        for expid in all_expids:
            dc = _extract_properties(expid)
            item = _finditem(dc, key)
            if item == value:
                currentids.add(expid)
        returnids = returnids.intersection(currentids)
    return returnids

def return_experiments(expids: Union[str,list,set]) -> dict:
    if isinstance(expids,str):
        expids = [expids]
    returndicts = {}
    for expid in expids:
        returndicts.update({expid : _extract_properties(expid)})
    return returndicts

def return_unique_values(key: str, reverse: bool = False) -> pd.Series:
    """
    Returns the unique values as keys, and the associated experiments as values
    experiment ids containing that value
    """
    dicts = {expid: _extract_properties(expid) for expid in all_expids}
    returndict = {}
    for expid, dc in dicts.items():
        item = _finditem(dc, key)
        if item is not None:
            try:
                expidlist = returndict[item]
                expidlist.append(expid)
            except KeyError: # If it is not yet present
                expidlist = [expid]
            returndict.update({item:expidlist})
    # Revamping to series
    returnseries = {value:pd.Series(ids) for value, ids in returndict.items()}
    returnseries = pd.concat(returnseries, axis = 0)
    returnseries.index.names = [key,'count']
    returnseries.name = 'expid'
    if reverse:
        returnseries_reversed = pd.Series(returnseries.index.get_level_values(0), index = pd.Index(returnseries.values, name = 'expid'))
        return returnseries_reversed
    else:
        return returnseries 


def load_pred_results(expid: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    predictor selection results as saved
    """
    resultspath = predselpath / f'{expid}_results.csv'
    df = pd.read_csv(resultspath, sep = ';', index_col = 0)
    df.index.name = 'npredictors'
    objecttypes = df.dtypes.loc[(df.dtypes == object).values]
    for col in objecttypes.index:
        if not col == 'cv_scores':
            f = lambda s: eval(s)
        else:
            f = lambda s: [float(f) for f in s[1:-1].split(' ') if f]
        df.loc[:,col] = df.loc[:,col].apply(f)
    cv_scores = pd.DataFrame(list(df['cv_scores'].values), index = df.index)
    cv_scores.columns.name = 'fold'
    return df.drop('cv_scores', axis = 1), cv_scores

def compare_scores(expids : Union[list,set], cv = True) -> pd.DataFrame:
    scoredfs = {}
    for expid in expids:
        try:
            df, cv_scores = load_pred_results(expid)
            if cv:
                scoredfs.update({expid:cv_scores})
            else:
                scoredfs.update({expid:df[['avg_score']]})
        except FileNotFoundError:
            warnings.warn(f'results file for {expid} not found. failed or still running. skipping to next')
    scoredf = pd.concat(scoredfs, axis = 0)
    scoredf.index.names = ['expid'] + scoredf.index.names[-1:]
    return scoredf

class BaseExplainer(object):
    def explain(self, X):
        """
        This aggregates into a global importance 
        but probably imperfectly so
        """
        self._attribute(X = X)
        return self.values.median(axis = 0)

class LinearModelExplainer(BaseExplainer):
    """
    use of coefficients. This overwrites the standard input*gradient
    input times gradient (i.e. coefficient)
    """
    def __init__(self, model):
        self.model = model

    def _attribute(self, X):
        coefs = self.model.coef_
        self.values = X * coefs[np.newaxis,:] 

    def explain(self, X):
        coefs = self.model.coef_
        return pd.Series(coefs, index = X.columns)
        
class TreeExplainer(shap.TreeExplainer, BaseExplainer):
    """
    Just a wrapper to define the method .explain()
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _attribute(self, X):
        shap_values = super().shap_values(X)
        self.values = pd.DataFrame(shap_values, index = X.index, columns = X.columns)

def return_explainer(model):
    """
    Return the right and initialized explainer object 
    """
    if isinstance(model, Pipeline):
        model = model[-1]
    if isinstance(model, tuple(_estimators[key] for key in ['linreg','ridreg'])):
        return LinearModelExplainer(model)
    elif isinstance(model, tuple(_estimators[key] for key in ['rfreg','rfclas','xgbreg'])):
        return TreeExplainer(model) 
    else: # probably rgresreg
        raise NotImplementedError(f'no explainer implemented for {type(model)}')

