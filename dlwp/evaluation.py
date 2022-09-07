import numpy as np
import pandas as pd
import xarray as xr
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from dlwp.remap import CubeSphereRemap

# library of units for each varlev used in DLWP
# use latex notation for plotting with matplotlib
varlev_units = {
    'z500':'m$^2$ s$^{-2}$',
    'z1000':'m$^2$ s$^{-2}$',
    'z250':'m$^2$ s$^{-2}$',
    't850':'K',
    't2m0':'K',
    'tau300-700':'m$^2$ s$^{-2}$',
    'tcwv0':'kg m$^{-2}$',
    'q850':'kg kg$^{-1}$',
    'u10m':'m s$^{-1}$',
    'v10m':'m s$^{-1}$',
}

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

def acc_cs(evaluation_directory, forecasts, plot_file):

    """
    
    """

    # initialize empty dicitonary to hold ACC. Keys will be var, values will
    # be lists with associated ACC for each forecast
    acc = {} 
    
    # iterate through forecasts and verify as specified in config  
    for forecast in forecasts: 
    
        print('getting ACC for configuration:')   
        print(forecast)
        
        for variable in forecast['variables']:
            
            f = ForecastEval(forecast_file=forecast['file'],
                             eval_variable=variable['varlev'],
                             cs_config=forecast['cs_config'],
                             unit_conversion=variable['unit_conversion'] if 'unit_conversion' in variable.keys() else None,
                             remap=False)
            print('generating verif...')
            f.generate_verification_from_predictor(**variable['verification_params'])
            print('done!')
            if variable['varlev'] not in acc.keys(): 
                acc[variable['varlev']]=[]
            acc[variable['varlev']].append({'f_hours': f.get_f_hour(),
                                            'acc':f.get_acc(),
                                            'plotting':forecast['plotting']})

    acc_plot(evaluation_directory, acc, plot_file)

def acc_ll(evaluation_directory, forecasts, plot_file):

    """
    
    """

    # initialize empty dicitonary to hold ACC. Keys will be var, values will
    # be lists with associated ACC for each forecast
    acc = {} 
    
    # iterate through forecasts and verify as specified in config  
    for forecast in forecasts: 
    
        print('getting ACC for configuration:')   
        print(forecast)
        
        for variable in forecast['variables']:
            
            f = ForecastEval(forecast_file=forecast['file'],
                             eval_variable=variable['varlev'],
                             cs_config=forecast['cs_config'],
                             unit_conversion=variable['unit_conversion'] if 'unit_conversion' in variable.keys() else None,
                             remap=True)
            print('generating verif...')
            f.generate_verification_from_era5(**variable['verification_params'])
            print('done!')
            if variable['varlev'] not in acc.keys(): 
                acc[variable['varlev']]=[]
            acc[variable['varlev']].append({'f_hours': f.get_f_hour(),
                                            'acc':f.get_acc(variable_name=variable['verification_params']['variable_name'],
                                                            level=variable['verification_params']['level'] if 'level' in variable['verification_params'].keys() else None),
                                            'plotting':forecast['plotting']})

    acc_plot(evaluation_directory, acc, plot_file)

def acc_plot(evaluation_directory, acc, plot_file):
            
    # create RMSE figure 
    ncols = int(np.ceil(len(acc.keys())/3))
    nrows = int(np.ceil(len(acc.keys())/ncols))         
    fig = plt.figure(figsize=(10*ncols,5*nrows))
    axs = []

    for i, variable in enumerate(acc.keys()):
         
        axs.append(fig.add_subplot(nrows,ncols,i+1)) # matplotlib uses 1-indexing
        axs[i].set_title(variable,fontsize=15) 
        axs[i].set_xlabel('Forecast Day',fontsize=15)
        axs[i].set_ylabel('ACC',fontsize=15)

        for var_acc in acc[variable]:
            axs[i].plot(var_acc['f_hours']/24,var_acc['acc'],**var_acc['plotting'])   

        axs[i].grid()
    
    axs[0].legend(fontsize=15)
    plt.tight_layout()
 
    plot_path = os.path.join(evaluation_directory,plot_file) 
    print('Saving plot to {}'.format(plot_path))
    fig.savefig(plot_path,dpi=300) 

def rmse_cs(evaluation_directory, forecasts, plot_file):

    """
    
    """

    # initialize empty dicitonary to hold RMSE. Keys will be var, values will
    # be lists with associated RMSE for each forecast
    rmse = {} 
    
    # iterate through forecasts and verify as specified in config  
    for forecast in forecasts: 
    
        print('getting RMSE for configuration:')   
        print(forecast)
        
        for variable in forecast['variables']:
            
            f = ForecastEval(forecast_file=forecast['file'],
                             eval_variable=variable['varlev'],
                             cs_config=forecast['cs_config'],
                             unit_conversion=variable['unit_conversion'] if 'unit_conversion' in variable.keys() else None,
                             remap=False)
            print('generating verif...')
            f.generate_verification_from_predictor(**variable['verification_params'])
            print('done!')
            if variable['varlev'] not in rmse.keys(): 
                rmse[variable['varlev']]=[]
            rmse[variable['varlev']].append({'f_hours': f.get_f_hour(),
                                             'rmse':f.get_rmse(),
                                             'plotting':forecast['plotting']})

    rmse_ll_plot(evaluation_directory, rmse, plot_file)

def rmse_ll(evaluation_directory, forecasts, plot_file):

    """
    
    """

    # initialize empty dicitonary to hold RMSE. Keys will be var, values will
    # be lists with associated RMSE for each forecast
    rmse = {} 
    
    # iterate through forecasts and verify as specified in config  
    for forecast in forecasts: 
    
        print('getting RMSE for configuration:')   
        print(forecast)
        
        for variable in forecast['variables']:
            
            f = ForecastEval(forecast_file=forecast['file'],
                             eval_variable=variable['varlev'],
                             cs_config=forecast['cs_config'],
                             unit_conversion=variable['unit_conversion'] if 'unit_conversion' in variable.keys() else None)
            print('generating verif...')
            getattr(f,variable['verification_generator'])(**variable['verification_params'])
            print('done!')
            if variable['varlev'] not in rmse.keys(): 
                rmse[variable['varlev']]=[]
            rmse[variable['varlev']].append({'f_hours': f.get_f_hour(),
                                             'rmse':f.get_rmse(),
                                             'plotting':forecast['plotting']})

    rmse_ll_plot(evaluation_directory, rmse, plot_file)

def rmse_ll_plot(evaluation_directory, rmse, plot_file):
            
    # create RMSE figure 
    ncols = int(np.ceil(len(rmse.keys())/3))
    nrows = int(np.ceil(len(rmse.keys())/ncols))         
    fig = plt.figure(figsize=(10*ncols,5*nrows))
    axs = []

    for i, variable in enumerate(rmse.keys()):
         
        axs.append(fig.add_subplot(nrows,ncols,i+1)) # matplotlib uses 1-indexing
        axs[i].set_title(variable,fontsize=15) 
        axs[i].set_xlabel('Forecast Day',fontsize=15)
        axs[i].set_ylabel('RMSE ({})'.format(varlev_units[variable]),fontsize=15)

        for var_rmse in rmse[variable]:
            axs[i].plot(var_rmse['f_hours']/24,var_rmse['rmse'],**var_rmse['plotting'])   

        axs[i].grid()
    
    axs[0].legend(fontsize=15)
    plt.tight_layout()
 
    plot_path = os.path.join(evaluation_directory,plot_file) 
    print('Saving plot to {}'.format(plot_path))
    fig.savefig(plot_path,dpi=300) 
            
        
class ForecastEval(object):
    
    """
    ForecastEval object is designed to take make forecast evaluation easier by 
    providing methods for analysis of a single field within a forecast. Support
    for information sharing accross object instances will allow for relatively 
    painless comparison of forecasts and evaluation of ensembles. 
    """
    
    def __init__(self, forecast_file=None, eval_variable=None, verbose=True, cs_config=None, unit_conversion=None, remap=True):
        
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
        :param unit_converstion: float: factor to convert units in forecast file 
        :param remap: bool: instructions whether to map CS forecast and predictor to LL
        """

        # check inputs
        if forecast_file is None or eval_variable is None:
            print('please provide a forecast file and evaluation variable')
            raise NameError

        # initialize verbosity
        self.verbose = verbose
 
        # initialize verification attributes 
        self.verification_file = None
        self.verification_file_LL = None
        self.verification_da = None
        self.verification_da_LL = None
        self.eval_var = eval_variable
        self.mse = None
        self._scaled = False
        self._valid_inits = None 

        # initialize forecast attributes and configuration variables
        self.forecast_file = forecast_file
        self.remap = remap
        self.forecast_da = None
        self.forecast_da_LL = None
        self.init_times = None
        self.forecast_range = None
        self.num_forecast_steps = None
        self.forecast_dt = None
        self.verification_range = None
        self.unit_conversion=unit_conversion
       
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
                if self.unit_conversion is None:
                    self.forecast_da = xr.open_dataset(forecast_file)[self.eval_var]
                else: 
                    print('converting values in forecast with by factor %s' % str(self.unit_conversion))
                    self.forecast_da = xr.open_dataset(forecast_file)[self.eval_var]*self.unit_conversion
                if self.remap is True:
                    if self.verbose:
                        print('Mapping forecast to a Lat-Lon mesh for evaluation')
                    # map to lat lon mesh and add proper coordinates
                    self.forecast_da_LL = convert_to_ll(self.forecast_da,
                                                        self.cs_config['path_to_remapper'],
                                                        self.cs_config['map_files'],
                                                        self.eval_var).assign_coords(time=self.forecast_da.time,
                                                                                     step=self.forecast_da.step)
                else: 
                    print("Intructed to not remap forecast into lat-lon mesh. Keeping forecast its CS format.")
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
        if self.remap:
            if self.verbose:
                print('remapping verification DataArray from the CS using config: %s' % str(self.cs_config))
            self.verification_da_LL = convert_to_ll(self.verification_da,
                                                    self.cs_config['path_to_remapper'],
                                                    self.cs_config['map_files'],
                                                    self.eval_var).assign_coords({'time':verif.time,
                                                                                  'step':verif.step})

    def generate_verification_from_era5(self, era5_file=None, variable_name=None, level=None,
                                        interpolation_method='linears'):
        """
        use raw EA5 data to create a verification field

        :param: string: era5_field:
            path and name of raw EA5 dataset in netcdf form
        :param: string: variable_name:
            name of evaluation variable within ERA5 dataset  above
        :param: float: level:
            level of evaluation variable
        :param: string: interpolation method:
            method used to interpolate era5 data to same mesh as forecast
        """
        # Check inputs
        if era5_file is None:
            print('Please indicate an ERA5 file')
            return
        # Assign file used in verification
        if self.verification_file_LL is not None:
            print('Replacing OLD verification dataset: %s' % self.verification_file_LL)
        self.verification_file_LL = era5_file
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

        self.verification_da_LL = verif

    def set_verification(self, verif):
        """
        set verification from incoming da. This is useful if comparing several forecasts from
        different models; you want to avoid generating the verification from data everytime.

        :param:  xarray.DataArray: verif:
            dataarray of formated verification 
        """
        self.verification_da_LL = verif

    def scale_das(self, scale_file=None, scale_verif=True):
        """
        Scale the incoming DataArray with mean and std defined in predictor file. This function will 
        change as we update scaling conventions.
        """
        if self._scaled is True:
            print('DataArrays have already been scaled')
        elif scale_file is not None:
            if self.verbose:
                print('Attempting to extract scaling statistics from scale file')
            # Attempt to extract the mean and std from verification file  
            mean = xr.open_dataset(scale_file)['mean'].values[0]
            std = xr.open_dataset(scale_file)['std'].values[0]
            if self.verbose:
                print('Scaling verification_da_LL and forecast_da_LL')           
            self.verification_da_LL = (self.verification_da_LL*std)+mean
            self.forecast_da_LL = (self.forecast_da_LL*std)+mean
            self._scaled = True

        else:
            if self.verbose:
                print('Attempting to extract scaling statistics from verif file')
            # Attempt to extract the mean and std from verification file  
            mean = xr.open_dataset(self.verification_file)['mean'].values[0]
            std = xr.open_dataset(self.verification_file)['std'].values[0]
            if self.verbose:
                print('Scaling forecast_da_LL by mean: %s and std: %s' % (str(mean),str(std)))           
            self.forecast_da_LL = (self.forecast_da_LL*std)+mean
            if scale_verif:
                if self.verbose:
                    print('Scaling verification_da_LL by mean: %s and std: %s' % (str(mean),str(std)))           
                self.verification_da_LL = (self.verification_da_LL*std)+mean
            self._scaled = True
    
    def get_mse_cs(self, mean=True):
        """
        return the MSE of the forecast against the verification on the Cuber sphere
        """
        # enforce proper order of dimensions in data arrays to avoid incorrect calculation
        f = self.forecast_da.transpose('step', 'time', 'face', 'height', 'width')
        verif = self.verification_da.transpose('step', 'time', 'face', 'height', 'width')
        # only calculate error over time available in verification
        valid_inits = self.verification_da.time
        # calculate mse
        if mean:
            return np.nanmean((verif.values-f.sel(time=valid_inits).values)
                              ** 2., axis=(1, 2, 3, 4))
        else:
            return np.nanmean((verif.values-f.sel(time=valid_inits).values)
                              ** 2., axis=(2, 3, 4))

    def get_rmse_cs(self, mean=True):
        """
        return the RMSE of forecast against verification
        """
        return np.sqrt(self.get_mse_cs(mean=mean))

    def get_mse(self, mean=True):
        """
        return the MSE of the forecast against the verification
        """
        if self.remap: 
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
        else:
            # enforce proper order of dimensions in data arrays to avoid incorrect calculation
            f = self.forecast_da.transpose('step', 'time', 'face', 'height', 'width')
            verif = self.verification_da.transpose('step', 'time', 'face', 'height', 'width')
            # only calculate error over time available in verification
            valid_inits = self.verification_da.time
            # calculate mse
            if mean:
                return np.nanmean((verif.values-f.sel(time=valid_inits).values)
                                  ** 2., axis=(1, 2, 3, 4))
            else:
                return np.nanmean((verif.values-f.sel(time=valid_inits).values)
                                  ** 2., axis=(2, 3, 4))
    
    def get_rmse(self, mean=True):
        """
        return the RMSE of forecast against verification
        """
        return np.sqrt(self.get_mse(mean=mean))
    
    def _daily_climo_time_series(self, climatology, times, step=None):
        """
        Generate a time series of daily climatology values from a climatology Dataset or DataArray and the specific
        desired times.

        :param climatology: xarray.Dataset or xarray.DataArray with a 'dayofyear' dimension
        :param times: iter: Timestamps
        :param step: iter: timedelta64[ns] or int
        :return: xarray.Dataset or xarray.DataArray with a 'time' dimension corresponding to daily climatologies for those
            times
        """
        if step is not None:
            result = []
            for s in step:
                doy = [pd.Timestamp(t + np.array(s).astype('timedelta64[ns]')).dayofyear for t in times]
                result.append(climatology.sel(dayofyear=doy).rename({'dayofyear': 'time'}).assign_coords(time=times))
            result = xr.concat(result, dim='step').assign_coords(step=step)
        else:
            doy = [t.dayofyear for t in times]
            result = climatology.sel(dayofyear=doy).rename({'dayofyear': 'time'}).assign_coords(time=times)
        return result

    def get_acc(self, climatology_file=None, mean=True, variable_name=None, level=None):
        
        if self.remap: 
            if climatology_file is None: 
                print('No provided climatology file, calculating from {}'.format(self.verification_file_LL))
                if variable_name is None:
                    print('Please provide variable name and level (if appropriate) for indexing of {}'.format(self.verification_file_LL))
                    return 
                if level is not None:
                    climatology = self._daily_climo_time_series(climatology = xr.open_dataset(self.verification_file_LL)[variable_name].sel(level=level).groupby('time.dayofyear').mean(),
                                                                times = self.verification_da_LL.time.values,
                                                                step=self.verification_da_LL.step.values)
                else:
                    climatology = self._daily_climo_time_series(climatology = xr.open_dataset(self.verification_file_LL)[variable_name].groupby('time.dayofyear').mean(),
                                                                times = self.verification_da_LL.time.values,
                                                                step=self.verification_da_LL.step.values)
                # transpose coordinates to match forecast and verification da 
                climatology = climatology.transpose('time','step','latitude','longitude')
                climatology = climatology.rename({'latitude':'lat','longitude':'lon'})
                climatology = climatology.assign_coords(self.verification_da_LL.coords) 
            else: 
                if level is None: 
                    climatology = xr.open_dataset(climotology_file)[variable_name]
                else: 
                    climatology = xr.open_dataset(climotology_file)[variable_name].sel(level=level)
            if mean:
                mean_over = (0,2,3)
                print('self.verification_da_LL: '+str(self.verification_da_LL))
                print('climatology: '+str(climatology))
                print('self.forecast_da_LL: '+str(self.forecast_da_LL))
                return (np.nanmean((self.verification_da_LL - climatology) * (self.forecast_da_LL - climatology), axis=mean_over)
                        / np.sqrt(np.nanmean((self.verification_da_LL- climatology) ** 2., axis=mean_over) *
                                  np.nanmean((self.forecast_da_LL - climatology) ** 2., axis=mean_over)))
            else:
                mean_over = (2,3)
                return (np.nanmean((self.verification_da_LL - climatology) * (self.forecast_da_LL - climatology), axis=mean_over)
                        / np.sqrt(np.nanmean((self.verification_da_LL - climatology) ** 2., axis=mean_over) *
                                  np.nanmean((self.forecast_da_LL - climatology) ** 2., axis=mean_over)))
        else:
            if climatology_file is None: 
                print('No provided climatology file, calculating from {}'.format(self.verification_file))
                climatology = self._daily_climo_time_series(climatology = xr.open_dataset(self.verification_file)['predictors'].groupby('sample.dayofyear').mean(),
                                                            times = self.verification_da.time.values,
                                                            step=self.verification_da.step.values)
            else: 
                climatology = xr.open_dataset(climotology_file)['predictors'].sel(varlev=self.eval_variable)
            if mean:
                mean_over = (0,2,3,4)
                return (np.nanmean((self.verification_da - climatology) * (self.forecast_da - climatology), axis=mean_over)
                        / np.sqrt(np.nanmean((self.verification_da- climatology) ** 2., axis=mean_over) *
                                  np.nanmean((self.forecast_da - climatology) ** 2., axis=mean_over)))
            else:
                mean_over = (2,3,4)
                return (np.nanmean((self.verification_da - climatology) * (self.forecast_da - climatology), axis=mean_over)
                        / np.sqrt(np.nanmean((self.verification_da- climatology) ** 2., axis=mean_over) *
                                  np.nanmean((self.forecast_da - climatology) ** 2., axis=mean_over)))

    def get_f_hour(self):
        """
        return an array with the leadtime of the forecast in hours 
        """
        f_hour = self.forecast_da.step/(3600*1e9)  # convert from nanoseconds to hours
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
