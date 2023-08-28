from utils import era5_retrieval, data_imputation
import numpy as np


era5_requests = [
    {'constant':True,
     'single_level_variable':True,
     'variable_name':'land_sea_mask',
     'grid':[1.,1.],
     'target_file':'/home/quicksilver2/nacc/Data/pipeline_dev/era5_1950-2022_3h_1deg_lsm.nc'},
    {'constant':False,
     'single_level_variable':True,
     'variable_name':'sst',
     'pressure_level':'1000',
     'grid':[1.,1.],
     'year': [y for y in range(1950,2023)],
     'month':[month+1 for month in range(0,12)],
     'day': [d+1 for d in range(0,31)],
     'time': np.arange(0,24,3).tolist(),
     'target_file':'/home/quicksilver2/nacc/Data/pipeline_dev/era5_1950-2022_3h_1deg_sst.nc'},
]
impute_params = {
    'FILENAME':'/home/quicksilver2/nacc/Data/pipeline_dev/era5_1950-2022_3h_1deg_sst.nc',
    'VARIABLE':'sst',
    'CHUNKS':{'time':10},
    'IMPUTED_FILE':'/home/quicksilver2/nacc/Data/pipeline_dev/era5_1950-2022_3h_1deg_sst-ti.nc'
}


# Retrive raw data
for request in era5_requests:
    era5_retrieval.main(request) 

# Impute ocean data 
data_imputation.triple_interp(impute_params)
