import numpy as np
import pandas as pd
import xarray as xr
import os
from tqdm import tqdm
from datetime import datetime
from dlwp import reamp

def convert_to_LL(self,da,path_to_remapper,map_files,var_name):
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
    csr = remap.CubeSphereRemap(path_to_remapper=path_to_remapper)
    csr.assign_maps(*map_files)

    #create netcdf from ds to go into remapper 
    da.load().to_netcdf('cs_data.nc')
    
    # map CS data to intermediate file
    csr.convert_from_faces('cs_data.nc','tmp_data.nc')
    # map to LL
    csr.inverse_remap('tmp_data.nc','ll_data.nc','--var',var_name)
    
    ll_ds = xr.open_dataset('ll_data.nc')[var_name].load()
    
    # clear working directory 
    os.remove('cs_data.nc')
    os.remove('tmp_data.nc')
    os.remove('ll_data.nc')

    return ll_ds

class ForecastEval(object):
    
    """
    ForecastEval is an object designed to painless evaluation of 
    forecasts produced by a DLWP model 
    """
    
    def __init__(self,forecast_file=None,eval_variable=None,verbose=True):
        
        """
        Initialize ForecastEval object. 
        
        :param forecast_file: string: 
               Path to forcast file. ForecastEval will attempt to initialize around
               the given forecast_file. If file does not already exist at this location, 
               any generated forecasts will be saved there. If None, path to generated
               forecast will be created automatically. 
        :param eval_variable: string:
              variable used to calculate evaluation metrics      
        """
        # initialize verbosity
        self.verbose=verbose
 
        # initialize verification attributes 
        self._verification_file = None
        self._verification_da = None
        self._verification_da_LL = None
        self._eval_var = eval_variable
        self._mse = None
        
        # initialize forecast attributes and configuration variables
        self._forecast_file = forecast_file  
        self._forecast_da = None
        self._forecast_da_LL = None
        self._init_times = None
        self._forecast_range = None
        self._num_forecast_steps = None
        self._forecast_dt = None
        self._verification_range = None
        
        # initialize forecast around file or configuration if given
        if self._forecast_file is not None: 
            
            if os.path.isfile(self._forecast_file):
                self._get_metadata_from_file()
                print('Initialized ForecastEval around file %s for %s' % (self._forecast_file,
                                                                          self._eval_var))
                self._forecast_da = xr.open_dataset(forecast_file)[self._eval_var]
            else: 
                print('%s was not found, ForecastEval was not able to initialize\
                       around a forecast.' % sel._forecast_file)        
    
    def _get_metadata_from_file(self):
        """"
        Attempt to extract metatdata from passed file 
        """ 
        file_ds = xr.open_dataset(self._forecast_file)
        
        self._init_times=file_ds.time.values
        self._forecast_range=file_ds.step[-1].values
        self._num_forecast_steps=len(file_ds.step)
        self._forecast_dt=file_ds.step[1].values-file_ds.step[0].values

    def generate_verification_from_predictor(self,predictor_file=None):
        """
        use predictor file to generate a verification ds 
        """
        # Check inputs 
        if predictor_file is None:
            print('Please indicate a predictor file')
            return

        # Find init dates that correspond to forecasted times sampled in the given predictor file
        valid_inits = self._find_valid_inits(xr.open_dataset(predictor_file).sample.values) 
        # Initialize array to hold verification data in forecast format 
        verif = xr.zeros_like(self._forecast_da.sel(time=valid_inits))*np.nan
        if self.verbose: 
            print('Generating verification dataset from predictor_file: %s' % predictor_file)
        for i in tqdm(range(len(valid_inits))):
            # create array of dates corresponding to forecast hours
            samples = pd.date_range(start=valid_inits[i],                             
                                    end=valid_inits[i]+(self._num_forecast_steps-1)*\
                                                            self._forecast_dt,             
                                    freq=pd.Timedelta(self._forecast_dt))                    
            # populate verif array with samples from date array above
            verif[i]=xr.open_dataset(predictor_file).predictors.sel(sample=samples).values.squeeze()
        self._verification_da = verif
        

    def _find_valid_inits(self,sample_array):
        """
        Find initializatio dates whose associated forecast samples are a subset of 
        sample_array
        
        param sample_array: np.array: array of datetime64 objects corresponding to 
             samples 
        """
        valid_inits = []
        for i in range(len(self._init_times)):
            samples = pd.date_range(start=self._init_times[i],                            
                                    end=self._init_times[i]+(self._num_forecast_steps-1)*
                                                            self._forecast_dt,           
                                    freq=pd.Timedelta(self._forecast_dt))                
            
            if np.all(np.in1d(samples,sample_array)):
                valid_inits.append(self._init_times[i])
            else: 
                if self.verbose:
                    print('forecast initialized at %s can not be verified with this predictor file; omitting.' % self._init_times[i])
        return valid_inits
