import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
try:
    import cartopy.crs as ccrs
except ImportError:
    pass

from typing import Union
from pathlib import Path

def plot_stations(stat: pd.Series, statloc: pd.DataFrame, fig = None, ax = None, cbar = True, scatter_kwargs = {}, cbar_kwargs = {'shrink':0.8}):
    try:
        s = scatter_kwargs.pop('s')
    except KeyError:
        s = 20
    if (fig is None):
        fig, ax = plt.subplots(subplot_kw={'projection':ccrs.PlateCarree()})
    lats = statloc.loc[stat.index,'LAT'].values
    lons = statloc.loc[stat.index,'LON'].values
    im = ax.scatter(x = lons, y = lats, c = stat.values, s= s, transform = ccrs.PlateCarree(), **scatter_kwargs)
    if cbar:
        fig.colorbar(im, ax = ax, **cbar_kwargs)
    return fig, ax, im

def plot_plus_stations(da: xr.DataArray, stat: pd.Series = None, statloc: pd.DataFrame = None,
        fig = None, ax = None, pcmesh_kwargs = {}, cbar = True, scatter_kwargs = {}):
    shading = 'flat'
    try:
        vmin = pcmesh_kwargs.pop('vmin')
    except KeyError:
        vmin = float(da.quantile(0.02))
    try:
        vmax = pcmesh_kwargs.pop('vmax')
    except KeyError:
        vmax = float(da.quantile(0.98))
    if (fig is None):
        fig, ax = plt.subplots(subplot_kw={'projection':ccrs.PlateCarree()})
    im = ax.pcolormesh(*data_for_pcolormesh(da, shading = shading), shading = shading, transform = ccrs.PlateCarree(),
            vmin = vmin, vmax = vmax, **pcmesh_kwargs)
    if not (stat is None):
        scatter_kwargs.update(dict(vmin = vmin, vmax = vmax, cmap = im.cmap))
        fig, ax, _ = plot_stations(stat = stat, statloc = statloc, fig = fig, ax = ax, cbar = False, scatter_kwargs = scatter_kwargs)
    if cbar:
        fig.colorbar(im, ax = ax, shrink = 0.8)
    return fig, ax, im
