from utils import era5_retrieval, data_imputation, map2hpx
from training.dlwp.data import data_loading as dl
import numpy as np
from omegaconf import OmegaConf

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
    'filename':'/home/quicksilver2/nacc/Data/pipeline_dev/era5_1950-2022_3h_1deg_sst.nc',
    'variable':'sst',
    'chunks':{'time':10000},
    'imputed_file':'/home/quicksilver2/nacc/Data/pipeline_dev/era5_1950-2022_3h_1deg_sst-ti.nc'
}
hpx_params = [
    {'file_name' : '/home/quicksilver2/nacc/Data/pipeline_dev/era5_1950-2022_3h_1deg_sst-ti.nc',
     'target_variable_name' : 'sst', 
     'file_variable_name' : 'sst', 
     'prefix' : '/home/quicksilver2/nacc/Data/pipeline_dev/era5_1deg_3h_HPX32_1950-2022_',
     'nside' : 32,
     'order' : 'bilinear', 
     'resolution_factor' : 1.0,
     'visualize':False},
    {'file_name' : '/home/quicksilver2/nacc/Data/pipeline_dev/era5_1950-2022_3h_1deg_lsm.nc',
     'target_variable_name' : 'lsm', 
     'file_variable_name' : 'lsm', 
     'prefix' : '/home/quicksilver2/nacc/Data/pipeline_dev/era5_1deg_3h_HPX32_1950-2022_',
     'nside' : 32,
     'order' : 'bilinear', 
     'resolution_factor' : 1.0,
     'visualize':False},
]
zarr_params = {
    'src_directory' : '/home/quicksilver2/nacc/Data/pipeline_dev/',
    'dst_directory' : '/home/quicksilver2/nacc/Data/pipeline_dev/',
    'dataset_name' : 'hpx32_1950-2021_3h_sst-only',
    'input_variables' : [
       'sst',],
    'output_variables' : [
        'sst',],
    'constants': {
        'lsm':'lsm'},
    'prefix' : 'era5_1deg_3h_HPX32_1950-2022_',
    'batch_size': 32,
    'scaling' : OmegaConf.load('/home/disk/quicksilver/nacc/data_pipeline/zephyr/training/configs/data/scaling/hpx32.yaml'),
    'overwrite' : False,
}
# Retrive raw data
for request in era5_requests:
    era5_retrieval.main(request) 

# Impute ocean data 
data_imputation.triple_interp(impute_params)

# Remap data to HPX mesh 
for hpx_param in hpx_params:
    map2hpx.main(hpx_param)

# create zarr file for optimized training 
dl.create_time_series_dataset_classic(**zarr_params)
