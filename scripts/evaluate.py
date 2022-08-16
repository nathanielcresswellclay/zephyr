import os 
import shutil
import argparse
import yaml
from  collections import defaultdict
import importlib

def evaluate(config):
    """
    evaluates a forecast as specified in config 

    param: config: dictionary: dict holding evaluation specifications
    """
    for analysis in config['analyses']:  
       
       module = importlib.import_module(analysis['module']) 
       getattr(module, analysis['method'])(config['evaluation_directory'],**analysis['params'])
    
    with open(os.path.join(config['evaluation_directory'],'config.yaml'),'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate Forecast[s]")
    parser.add_argument('config', type=str)
    parser.add_argument('--overwrite', action='store_true')
    run_args = parser.parse_args()

    with open(run_args.config, 'r') as stream: 
        config = yaml.safe_load(stream)
    
    if os.path.isdir(config['evaluation_directory']) and not run_args.overwrite:
        print('Evaluation directory {} already exists. Aborting evaluation, to \
overwrite use flag "--overwrite"'.format(config['evaluation_directory']))
    else:
        try:
            shutil.rmtree(config['evaluation_directory'])
            print('overwriting previous evaluation in {}'.format(config['evaluation_directory']))
        except FileNotFoundError:
            print('creating directory {} to store evaluation'.format(config['evaluation_directory'])) 
        os.mkdir(config['evaluation_directory'])
        evaluate(config) 
    
