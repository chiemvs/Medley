import os
import sys
import json
import numpy as np
import pandas as pd
import pyarrow as pa

from pathlib import Path
from typing import Union

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

def return_unique_values(key: str) -> list:
    """
    Returns the unique values for a key, plus a list of
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
    return returndict


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
        df, cv_scores = load_pred_results(expid)
        if cv:
            scoredfs.update({expid:cv_scores})
        else:
            scoredfs.update({expid:df[['avg_score']]})
    scoredf = pd.concat(scoredfs, axis = 0)
    return scoredf
