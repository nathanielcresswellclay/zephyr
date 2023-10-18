from dask.diagnostics import ProgressBar
import xarray as xr
import forecast
import copy
import os


# helper class to create attributable object from dictionary 
# this is necessary for inference treatment of parameters 
class ParamDict(object):
    def __init__(self, d):
        self.__dict__ = d

def ensemble_inference(models, params, ensemble_params, overwrite):
    """
    This function will create an ensemble forecast from the model-checkpoint combinations listed 
    in models. 

    param models: sequence: list of dictionaries that have a model directory path and a checkpoint 
          leaving checkpoint as None with select the checkpoint with the least validation loss 
    param params: dictionary: passed to interence to create forecast. These parameters are consistent
          between model-checkpoint combinations
    param overwrite: boolean: if existing forecast files should be overwritten
    """
    import json 
    #pretty=json.dumps(params,indent=2)
    #print(pretty)

    # list of params to send to inference 
    param_list = []
    # populate params list with model specific information
    for model in models: 
        model_params = copy.deepcopy(params)
        model_params['model_path'] = model['model_path']
        model_params['model_checkpoint'] = model['checkpoint']
        model_params['output_filename'] = model['output_filename']
        model_params['hydra_path'] = os.path.relpath(model_params["model_path"], f'{os.getcwd()}')
        param_list.append(model_params)
        del model_params
          

    # create member forecasts 
    for model_param in param_list:
        if os.path.isfile(f'{model_param["output_directory"]}/{model_param["output_filename"]}.nc') and not overwrite:
            print(f'file: {model_param["output_directory"]}/{model_param["output_filename"]}.nc already exists.')
            print(f'To replace existing file pass "overwrite=True"')
        else:
            # perform inference
            forecast.inference(ParamDict(model_param))
    
    # create ensemble 
    member_forecasts = [f"{m['output_directory']}/{m['output_filename']}.nc" for m in param_list]
    ensemble_forecast = xr.open_mfdataset(member_forecasts, concat_dim='ensemble_member',combine='nested')
    # write ensemble file
    if os.path.isfile(f"{ensemble_params['output_directory']}/{ensemble_params['output_filename']}.nc") and not overwrite:
        print(f"{ensemble_params['output_directory']}/{ensemble_params['output_filename']}.nc already exists")
        print(f'To replace existing file pass "overwrite=True"')
    else:
        print(f"saving ensemble forecast to {ensemble_params['output_directory']}/{ensemble_params['output_filename']}.nc")
        with ProgressBar():
            ensemble_forecast.to_netcdf(f"{ensemble_params['output_directory']}/{ensemble_params['output_filename']}.nc")
    
     

#########################################################################################
####################################    EXAMPLE    ######################################
#########################################################################################

# list of model directories, include full path 
MODELS = [
    {'model_path':'/home/disk/quicksilver/nacc/S2S/zephyr/outputs/hpx64_unet_136-68-34_cnxt_skip_dil_gru_6h_300_seed0',
     'checkpoint': None,
     'output_filename':'test1'},
    {'model_path':'/home/disk/quicksilver/nacc/S2S/zephyr/outputs/hpx64_unet_136-68-34_cnxt_skip_dil_gru_6h_300_seed1',
     'checkpoint': None,
     'output_filename':'test2'},
]
# parameters passed to inference, the same parameters are applied to all models 
FORECAST_PARAMS = {
    'model_version' : None,
    'non_strict' : False,
    'lead_time' : 336,
    'forecast_init_start' : '2017-01-02',
    'forecast_init_end' : '2018-12-30',
    'freq' : 'biweekly',
    'output_directory' : '/home/disk/quicksilver/nacc/S2S',
    'encode_int' : False,
    'to_zarr' : False,
    'data_directory' : '/home/quicksilver2/karlbam/Data/DLWP/HPX64',
    'data_prefix' : 'era5_hpx64_7var_6h_24h',
    'data_suffix' : None,
    'gpu' : 0,
    'batch_size':None,
}
# parameters useful for calcluating ensemble forecasts
ENSEMBLE_PARAMS = {
    'output_directory' : '/home/disk/quicksilver/nacc/S2S',
    'output_filename': 'test_ensemble',
}
OVERWRITE = False

if __name__ == "__main__" :

    ensemble_inference(MODELS, FORECAST_PARAMS, ENSEMBLE_PARAMS, OVERWRITE)
     
    
