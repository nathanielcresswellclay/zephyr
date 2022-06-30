import numpy as np
import pandas as pd
import xarray as xr
import os
from tqdm import tqdm
from dlwp.remap import CubeSphereRemap


def convert_to_ll(da, path_to_remapper, map_files, var_name):
    """
    Map ds from CS to LL mesh

    param da: DataArray:
         array to be converted to Lat-Lon mesh  
    param path_to_remapper: str:
         location of tempest remap utilities
    param map_files: list:
         list of map files to use in remapping 
    param var_name: str:
         variable name to use when applying inverse map. This is the 
         variable name in the ds 
    """
    # initialize CSR object 
    csr = CubeSphereRemap(path_to_remapper=path_to_remapper)
    csr.assign_maps(*map_files)

    # create netcdf from ds to go into remapper
    da.load().to_netcdf('cs_data.nc')
    
    # map CS data to intermediate file
    csr.convert_from_faces('cs_data.nc', 'tmp_data.nc')
    # map to LL
    csr.inverse_remap('tmp_data.nc', 'll_data.nc')
    
    ll_da = xr.open_dataset('ll_data.nc')[var_name].load()
    
    # clear working directory 
    os.remove('cs_data.nc')
    os.remove('tmp_data.nc')
    os.remove('ll_data.nc')

    return ll_da


class ForecastEval(object):
    
    """
    ForecastEval is an object designed to painless evaluation of forecasts produced by a DLWP model
    """
    
    def __init__(self, forecast_file=None, eval_variable=None, verbose=True, cs_config=None):
        
        """
        Initialize ForecastEval object. 
        
        :param forecast_file: string: 
               Path to forcast file. ForecastEval will attempt to initialize around
               the given forecast_file. If file does not already exist at this location, 
               any generated forecasts will be saved there. If None, path to generated
               forecast will be created automatically. 
        :param eval_variable: string:
              variable used to calculate evaluation metrics      
        :param cs_config: dict:
               a dictionary that configures the remap of the forecast and verification
               from the CubeSphere 
        """

        # check inputs
        if forecast_file is None or eval_variable is None:
            print('please provide a forecast file and evaluation variable')
            raise NameError

        # initialize verbosity
        self.verbose = verbose
 
        # initialize verification attributes 
        self.verification_file = None
        self.verification_da = None
        self.verification_da_LL = None
        self.eval_var = eval_variable
        self.mse = None
        self._scaled = False
        self._valid_inits = None 

        # initialize forecast attributes and configuration variables
        self.forecast_file = forecast_file
        self.forecast_da = None
        self.forecast_da_LL = None
        self.init_times = None
        self.forecast_range = None
        self.num_forecast_steps = None
        self.forecast_dt = None
        self.verification_range = None
       
        # initialize configuration of CubeSphereRemap
        if cs_config is None:
            self.cs_config = {'path_to_remapper': '/home/disk/brume/nacc/tempestremap/bin',
                              'map_files': ('/home/disk/brass/nacc/map_files/O1/map_LL121x240_CS64.nc',
                                            '/home/disk/brass/nacc/map_files/O1/map_CS64_LL121x240.nc')}
        else:
            self.cs_config = cs_config

        # initialize forecast around file or configuration if given
        if self.forecast_file is not None:
            
            if os.path.isfile(self.forecast_file):
                self._get_metadata_from_file()
                if self.verbose:
                    print('Initialized ForecastEval around file %s for %s' % (self.forecast_file,
                                                                              self.eval_var))
                self.forecast_da = xr.open_dataset(forecast_file)[self.eval_var]
                if self.verbose:
                    print('Mapping forecast to a Lat-Lon mesh for evaluation')
                # map to lat lon mesh and add proper coordinates
                self.forecast_da_LL = convert_to_ll(self.forecast_da,
                                                    self.cs_config['path_to_remapper'],
                                                    self.cs_config['map_files'],
                                                    self.eval_var).assign_coords(time=self.forecast_da.time,
                                                                                 step=self.forecast_da.step)
            else: 
                print('%s was not found, ForecastEval was not able to initialize\
                       around a forecast.' % forecast_file)
    
    def generate_verification_from_predictor(self, predictor_file=None):
        """
        use predictor file to generate a verification ds

        :param: string: predictor_file:
              File with path to predictor file around which a verification dataset will be 
              generated. 
        """
        # Check inputs 
        if predictor_file is None:
            print('Please indicate a predictor file')
            return
        # Assign file used in verification
        if self.verification_file is not None:
            print('Replacing OLD verification dataset: %s' % self.verification_file)
        self.verification_file = predictor_file
        # Find init dates that correspond to forecasted times sampled in the given predictor file
        valid_inits = self._find_valid_inits(xr.open_dataset(predictor_file).sample.values) 
        # Initialize array to hold verification data in forecast format 
        verif = xr.zeros_like(self.forecast_da.sel(time=valid_inits))*np.nan
        if self.verbose: 
            print('Generating verification dataset from predictor_file: %s' % predictor_file)
        for i in tqdm(range(len(valid_inits))):
            # create array of dates corresponding to forecast hours
            samples = pd.date_range(start=valid_inits[i],
                                    end=valid_inits[i]+(self.num_forecast_steps-1) *
                                    self.forecast_dt,
                                    freq=pd.Timedelta(self.forecast_dt))
            # populate verif array with samples from date array above
            verif[i] = xr.open_dataset(predictor_file).predictors.sel(sample=samples).values.squeeze()
        self.verification_da = verif
        if self.verbose:
            print('remapping verification DataArray from the CS using config: %s' % str(self.cs_config))
        self.verification_da_LL = convert_to_ll(self.verification_da,
                                                self.cs_config['path_to_remapper'],
                                                self.cs_config['map_files'],
                                                self.eval_var)

    def generate_verification_from_era5(self, era5_file=None, variable_name=None, level=None, conversion=None,
                                        interpolation_method='linears'):
        """
        use raw EA5 data to create a verification field

        :param: string: era5_field:
            path and name of raw EA5 dataset in netcdf form
        :param: string: variable_name:
            name of evaluation variable within ERA5 dataset  above
        :param: float: level:
            level of evaluation variable
        :param: float: conversion:
            unit conversion factor
        :param: string: interpolation method:
            method used to interpolate era5 data to same mesh as forecast
        """
        # Check inputs
        if era5_file is None:
            print('Please indicate a ERA5 file')
            return
        # Assign file used in verification
        if self.verification_file is not None:
            print('Replacing OLD verification dataset: %s' % self.verification_file)
        self.verification_file = era5_file
        # Find init dates that correspond to forecasted times sampled in the given predictor file
        valid_inits = self._find_valid_inits(xr.open_dataset(era5_file).time.values)
        # Initialize array to hold verification data in forecast format
        verif = xr.zeros_like(self.forecast_da_LL.sel(time=valid_inits))*np.nan
        if self.verbose:
            print('Generating verification dataset from era5_file: %s' % era5_file)

        for i in tqdm(range(len(valid_inits))):
            # create array of dates corresponding to forecast hours
            samples = pd.date_range(start=valid_inits[i],
                                    end=valid_inits[i] + (self.num_forecast_steps - 1) * self.forecast_dt,
                                    freq=pd.Timedelta(self.forecast_dt))
            # populate verif array with samples from date array above
            if level is not None:
                tmp = xr.open_dataset(era5_file)[variable_name].sel(time=samples, level=level)
            else:
                tmp = xr.open_dataset(era5_file)[variable_name].sel(time=samples)
            verif[i] = tmp.values.squeeze()
#            verif[i] = tmp.interp(latitude=self.forecast_da_LL.lat, longitude=self.forecast_da_LL,
#                                  method=interpolation_method).values.squeeze()

        if conversion is not None:
            print('converting verification array using %s factor' % str(conversion))
            verif = verif*conversion

        self.verification_da_LL = verif

    def set_verification(self, verif):
        """
        set verification from incoming da. This is useful if comparing several forecasts from
        different models; you want to avoid generating the verification from data everytime.

        :param:  xarray.DataArray: verif:
            dataarray of formated verification 
        """
        self.verification_da_LL = verif

    def scale_das(self):
        """
        Scale the incoming DataArray with mean and std defined in predictor file. This function will 
        change as we update scaling conventions.
        """
        if self._scaled is True:
            print('DataArrays have already been scaled')
        else:
            if self.verbose:
                print('Attempting to extract scaling statistics from verif file')
            # Attempt to extract the mean and std from verification file  
            mean = xr.open_dataset(self.verification_file)['mean'].values[0]
            std = xr.open_dataset(self.verification_file)['std'].values[0]
            if self.verbose:
                print('Scaling verification_da_LL and forecast_da_LL')           
            self.verification_da_LL = (self.verification_da_LL*std)+mean
            self.forecast_da_LL = (self.forecast_da_LL*std)+mean
            self._scaled = True

    def get_mse(self, mean=True):
        """
        return the MSE of the forecast against the verification
        """
        # calculate the weights to apply to the MSE
        weights = np.cos(np.deg2rad(self.verification_da_LL.lat.values))
        weights /= weights.mean()        
        # enforce proper order of dimensions in data arrays to avoid incorrect calculation
        f = self.forecast_da_LL.transpose('step', 'time', 'lon', 'lat')
        verif = self.verification_da_LL.transpose('step', 'time', 'lon', 'lat')
        # reshape weights to be compatible with verif array
        weights = np.expand_dims(np.expand_dims(np.expand_dims(weights, axis=0), axis=0), axis=0)
        # only calculate error over time available in verification
        valid_inits = self.verification_da_LL.time
        # calculate mse
        if mean:
            return np.nanmean((verif.values-f.sel(time=valid_inits).values)
                              ** 2. * weights, axis=(1, 2, 3))
        else:
            return np.nanmean((verif.values-f.sel(time=valid_inits).values)
                              ** 2. * weights, axis=(2, 3))

    def get_rmse(self, mean=True):
        """
        return the RMSE of forecast against verification
        """
        return np.sqrt(self.get_mse(mean=mean))
    
    def get_f_hour(self):
        """
        return an array with the leadtime of the forecast in hours 
        """
        f_hour = self.forecast_da_LL.step/(3600*1e9)  # convert from nanoseconds to hours
        f_hour = np.array(f_hour, dtype=float)
        return f_hour

    def _get_metadata_from_file(self):
        """
        Attempt to extract metatdata from passed file 
        """ 
        file_ds = xr.open_dataset(self.forecast_file)
        
        self.init_times = file_ds.time.values
        self.forecast_range = file_ds.step[-1].values
        self.num_forecast_steps = len(file_ds.step)
        self.forecast_dt = file_ds.step[1].values-file_ds.step[0].values

    def _find_valid_inits(self, sample_array):
        """
        Find initializatio dates whose associated forecast samples are a subset of 
        sample_array
        
        param sample_array: np.array: array of datetime64 objects corresponding to 
             samples 
        """
        valid_inits = []
        for i in range(len(self.init_times)):
            samples = pd.date_range(start=self.init_times[i],
                                    end=self.init_times[i]+(self.num_forecast_steps-1) *
                                    self.forecast_dt,
                                    freq=pd.Timedelta(self.forecast_dt))
            
            if np.all(np.in1d(samples, sample_array)):
                valid_inits.append(self.init_times[i])
            else: 
                if self.verbose:
                    print('forecast initialized at %s can not be verified with this predictor file; omitting.'
                          % self.init_times[i])
        return valid_inits
