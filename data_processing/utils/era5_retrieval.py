import xarray as xr
import cdsapi 
import os
import numpy as np
from dask.diagnostics import ProgressBar
import copy


EXAMPLE = {
    'constant':False,
    'single_level_variable':False,
    'variable_name':'temperature',
    'pressure_level':'1000',
    'grid':[1.,1.],
    'year': [2000],   #[y for y in range(1979,2023)],
    'month':[1],      #[month+1 for month in range(0,12)],
    'day': [1],       #[d+1 for d in range(0,31)],
    'time': np.arange(0,24,3).tolist(),
    'target_file':'test_t1000.nc'
}
EXAMPLE_CONSTANT = {
    'constant':True,
    'single_level_variable':True,
    'variable_name':'land_sea_mask',
    'grid':[1.,1.],
    'target_file':'/home/quicksilver2/nacc/Data/pipeline_dev/era5_1979-2022_3h_1deg_lsm.nc',
}

def estimate_num_requests(request_dict):
    """
    A helper function for compiling requests. CDS only accepts requests of 120000 
    "items" where item is a time step of one variable. Here were calculate how 
    many requests we will need for a given requests dictionary 
    """ 
    items = len(request_dict['time']) *\
            len(request_dict['day']) *\
            len(request_dict['month']) *\
            len(request_dict['year'])
    return(int(np.ceil(items/120000)))
    
def partition_requests(n_requests, requests_dict):
    """
    If the request needs to be broken up (n_requests>1) then splits by year
    """ 
    if n_requests < 2: 
        return [requests_dict]
    else:
        requests = []
        parts = np.array_split(np.array(requests_dict['year'],dtype=str),n_requests)
        for part in parts:
            requests.append(copy.deepcopy(requests_dict))
            requests[-1]['year'] = part.tolist()
        return requests
 

def compile_request(params): 
    """
    Turns param dictionary into proper format for cdsapi requests, accounts for
    cases of constant field or time varying field 
    """
    
    if params['constant']:
        requests_dict = {
            'variable': params['variable_name'],
            'product_type': 'reanalysis',
            'grid':params['grid'],
            'year': 2000, #   
            'month': 1,   #  Dummy values required   
            'day': 1,     #  to complete request 
            'time': 0,    #         
            'format': 'netcdf'}
        return [requests_dict]

    else:
        requests_dict = {
            'variable': params['variable_name'],
            'pressure_level': params['pressure_level'],
            'product_type': 'reanalysis',
            'grid':params['grid'],
            'year': params['year'],
            'month': params['month'],
            'day': params['day'],
            'time': params['time'],
            'format': 'netcdf'}
        num_requests = estimate_num_requests(requests_dict)
        return partition_requests(num_requests,requests_dict)
    

def main(params):
    """
    Takes params from incoming dictionary, formats requests to cdsapi call and fetches
    ERA5 data. Resolves partitioning of large requests. 

    params: 
    - params :dictionary: identifies the data to be fetched from cds. variable and 
      constant fields are supported with seperate data streams. Format of dictionary
      is as follows for variable and consant fields:

          EXAMPLE_VARIABLE = {
          
              'constant':False,
              'single_level_variable':False,
              'variable_name':'temperature',
              'pressure_level':'1000',
              'grid':[1.,1.],
              'year': [2000],   #[y for y in range(1979,2023)],
              'month':[1],      #[month+1 for month in range(0,12)],
              'day': [1],       #[d+1 for d in range(0,31)],
              'time': np.arange(0,24,3).tolist(),
              'target_file':'test_t1000.nc'
          }
          EXAMPLE_CONSTANT = {
              'constant':True,
              'single_level_variable':True,
              'variable_name':'land_sea_mask',
              'grid':[1.,1.],
              'target_file':'/home/quicksilver2/nacc/Data/pipeline_dev/era5_1979-2022_3h_1deg_lsm.nc',
          }
    """
    
    if os.path.isfile(params['target_file']):
        print(f'File already exists at location: {params["target_file"]}. \
If issue arrises delete file and try again.')
    else:
        c = cdsapi.Client()
        variable_type = 'reanalysis-era5-single-levels' if params['single_level_variable']\
                         else 'reanalysis-era5-pressure-levels'
        requests = compile_request(params)
        file_partitions = []
        for i,request in enumerate(requests): 
            c.retrieve(variable_type,
                       request,
                       params['target_file']+f'.p{i}',)
            file_partitions.append(params['target_file']+f'.p{i}')
        
        print(f'chunking and concatenating (if necessary) retrieved files')
        concated_ds = xr.open_mfdataset(file_partitions,
                                        combine = 'nested',
                                        chunks=dict(time=10) if not params['constant'] else None,
                                        concat_dim='time' if not params['constant'] else None)
        if params['constant']:
            concated_ds = concated_ds.squeeze()
        delayed_write = concated_ds.to_netcdf(params['target_file'],compute=False)
        print(f'writing to disk...')
        with ProgressBar():
            delayed_write.compute()
        print(xr.open_dataset(params['target_file']))
        for f in file_partitions:
            os.remove(f)
            
        

if __name__=="__main__":
    
    main(EXAMPLE_PARAMS)
