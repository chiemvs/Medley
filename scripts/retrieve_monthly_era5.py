import zarr
import numpy as np
import xarray as xr
import cdsapi

from pathlib import Path

store = 'reanalysis-era5-pressure-levels-monthly-means'
tempdir = Path('/scistor/ivm/jsn295/temp/')
finalpath = Path('/scistor/ivm/jsn295/Medi/monthly/monthly_u20_era5.zarr')

def request_one_year(year: int):
    request = {'format':'netcdf',
               'product_type': 'monthly_averaged_reanalysis',
               'variable': 'u_component_of_wind',
               'pressure_level': '20',
               'year':str(year),
               'month':[f'0{i}'[-2:] for i in range(1,13)],
               'time':'00:00',
               'area':[90, -180, -20,180],
               }
    return request

if __name__ == '__main__':
    c = cdsapi.Client()
    for year in range(1940,2023):
        temppath = tempdir / f'u20_{year}.nc'
        if not temppath.exists():
            c.retrieve(store, request_one_year(year), temppath)
    
    # Reading and writing to zarr
    if not finalpath.exists():
        complete = xr.open_mfdataset(list(tempdir.glob('u20_*.nc')))
        complete['u'] = complete['u'].chunk({'time':len(complete.time),'latitude':50, 'longitude':50})
        complete.to_zarr(finalpath)
        
