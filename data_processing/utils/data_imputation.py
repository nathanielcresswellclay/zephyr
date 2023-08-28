import xarray as xr 
from dask.diagnostics import ProgressBar
import numpy as np 

# params for triple interpolation imputation of sst 
PARAMS_TI = {
    'FILENAME':'/home/disk/brass/nacc/data/ERA5/1-deg/1979-2021_era5_1deg_3h_sea_surface_temperature.nc',
    'VARIABLE':'sst',
    'CHUNKS':{'time':10},
    'IMPUTED_FILE':'/home/rhodium/dlwp/data/era5/1deg/1979-2021_era5_1deg_3h_sea_surface_temperature_ti-imputed.nc'
}

# params for zonal climotological imputation of sst 
PARAMS_CLIMO = {
    'FILENAME':'/home/rhodium/dlwp/data/era5/1deg/1979-2021_era5_1deg_3h_sea_surface_temperature.nc',
    'VARIABLE':'sst',
    'CHUNKS':{'time':10},
    'IMPUTED_FILE':'/home/rhodium/dlwp/data/era5/1deg/1979-2021_era5_1deg_3h_sea_surface_temperature_climo-imputed.nc'
}

def zonal_climo_impute(params):

    da = xr.open_dataset(params['FILENAME'],chunks = params['CHUNKS'])[params['VARIABLE']]
    # calculate climotology 
    climo = xr.open_dataset(params['FILENAME'],chunks = params['CHUNKS'])[params['VARIABLE']].mean(dim='time')
    # get zonal climatology and expand dimensions so that it is defined in lat-lon dimensions  
    zonal_climo = climo.mean(dim='longitude').expand_dims(dim={'longitude':climo.longitude.values},axis=1)
    # use polar interpolation to fill in polar NaNs
    zonal_climo_polar_interp = zonal_climo.roll(latitude=int(len(zonal_climo.latitude)/2)).interpolate_na(dim='latitude',method='linear',use_coordinate=False).roll(latitude=len(zonal_climo.latitude)-int(len(zonal_climo.latitude)/2))
    # impute NaN pixels with new polar interpolated  zonally climatology 
    da_climo_imputed = xr.where(np.isnan(da),zonal_climo_polar_interp,da)
     
    print(f'Beginning computation of climotological imputation and writing...')
    print(f'INPUT FILENAME: {params["FILENAME"]}')
    print(f'IMPUTED FILENAME: {params["IMPUTED_FILE"]}') 
    with ProgressBar():
        result = da_climo_imputed.to_netcdf(params['IMPUTED_FILE']) 
    if result is None: 
        print(f'Processes exited successfully. Hooray!')

def triple_interp(params):

    da = xr.open_dataset(params['FILENAME'],chunks = params['CHUNKS'])[params['VARIABLE']]
    # first go at interpolation along lines of latitude 
    da_interp = da.interpolate_na(dim='longitude',method='linear',use_coordinate=False)
    # offset and interpolate again, this allows interpolation where nans persist to boundary 
    da_double_interp = da_interp.roll(longitude=int(len(da_interp.longitude)/2)).interpolate_na(dim='longitude',method='linear',use_coordinate=False).roll(longitude=len(da_interp.longitude)-int(len(da_interp.longitude)/2))
    # last round of interpolation fills in N-S boundary nans by offsetting and interpolating pole to pole. 
    da_triple_interp = da_double_interp.roll(latitude=int(len(da_interp.latitude)/2)).interpolate_na(dim='latitude',method='linear',use_coordinate=False).roll(latitude=len(da_interp.latitude)-int(len(da_interp.latitude)/2))
    
    print(f'Beginning computation of interpolation and writing...')
    print(f'INPUT FILENAME: {params["FILENAME"]}')
    print(f'IMPUTED FILENAME: {params["IMPUTED_FILE"]}') 
    with ProgressBar():
        result = da_triple_interp.to_netcdf(params['IMPUTED_FILE']) 
    if result is None: 
        print(f'Processes exited successfully. Hooray!')
    

if __name__ == "__main__" :

    triple_interp(params)
    zonal_climo_impute(params)
