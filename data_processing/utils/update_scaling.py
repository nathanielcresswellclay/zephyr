import numpy as np 
import xarray as xr
from omegaconf import OmegaConf, open_dict
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)

EXAMPLE_PARAMS = {
    'scale_file' : '/home/disk/quicksilver/nacc/dlesm/zephyr/training/configs/data/scaling/hpx32.yaml',
    'variable_file_prefix' : '/home/disk/rhodium/dlwp/data/HPX32/era5_1deg_3h_HPX32_1950-2022_',
    'variable_names' : ['z1000-24H'],
    'selection_dict' : {'sample':slice(np.datetime64('2022-12-21'),np.datetime64('2022-12-31'))},
    'overwrite' : False,
    'chunks' : None,
}

def main(params):
    """
    Update the scaling parameters for a list of variables in a dataset.

    Parameters:
    params (dict): A dictionary with the following keys:
        - 'scale_file': Path to the YAML file containing the scaling parameters.
        - 'variable_file_prefix': Prefix for the file names containing the variables.
        - 'variable_names': List of variable names to update.
        - 'selection_dict': Dictionary defining the data subset to use for the calculation.
        - 'overwrite': Whether to overwrite existing scaling parameters.
        - 'chunks': Dictionary defining the chunk sizes for the data loading.
    """

    # Load the current scaling parameters
    scale_dict = OmegaConf.to_container(OmegaConf.load(params['scale_file']))

    # Loop over the variable names
    for variable_name in params['variable_names']:
        # Check if the variable is already in the scale file
        if variable_name in scale_dict.keys() and not params['overwrite']:
            logging.info(f"{variable_name} is already in {params['scale_file']}. If you wish to overwrite existing value, pass 'overwrite'=True")
            continue
        else: 
            logging.info(f"Adding {variable_name} to {params['scale_file']}")

        # Load the data for the variable
        da = xr.open_dataset(params['variable_file_prefix'] + variable_name + '.nc',chunks=params['chunks'])[variable_name].sel(params['selection_dict'])

        # Calculate the mean and standard deviation
        if params['chunks'] is None:
            da.load()
            mean = da.values.mean()
            std = da.values.std()
        else:
            raise NotImplementedError("chunked calculation of mean and std is not yet implemented. Xarray's built-in std function is bad!")

        # Update the scaling parameters for the variable
        scale_dict[variable_name] = { 'mean':mean.item(),'std':std.item() }

    # Save the updated scaling parameters
    scale_conf = OmegaConf.create(scale_dict)  
    OmegaConf.save(config = scale_conf, f=params['scale_file'])

if __name__ == "__main__":
    # Run the main function with the example parameters
    main(EXAMPLE_PARAMS)
    
    
    


