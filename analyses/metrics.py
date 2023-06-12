import matplotlib.pyplot as plt
import os
import numpy as np
from ..evaluation import evaluators
from .statics import variable_labels as vl


def get_daily_interval(fhours):

    """
    This function takes a list of forecast hours and subsampl
    """
    
    return np.arange(fhours[0],fhours[-1]+1,24)

def acc_rmse_compare(
        forecasts: list, eval_variables: list, plot=True, plot_file='metrics.pdf',
        xticks=[0,24,48,72,96,120,144,168,192,216,240,264,288,312,336],):           
    """
    Use this function to compare the ACC and RMSE of different models 
    accross one or more variables 
    
    PARAMETERS:
    - forecasts (list): a list of dictionaries. Each dictionary is 
      associated with a forecast to evaluate. The dictionaries contain 
      instructions on how to evaluate the associated forecast. Example 
      dictionary: 

      {'filename':'/path/to/forecast.nc', # absolute path to forecast 
           # file 
       'ground_truth':{
           'z500':'/path/to/gt_z500.nc',
           'z1000':'/path/to/gt_z1000.nc',} # dictionary in which 
           # keys are eval_variables and values are absolute paths 
           # to the verification datasets 
       'label': 'model-identifier', # label to identify this model in 
           # plot legends and as a key in metrics dictionary 
       'evaluator_kwargs':{} # a dictionary of kwargs to pass to 
           # evaluator constructor
       'climatology_forecast_prefix':'/path/to/climo_ # this is the 
           # prefix upon which the variable name and '.nc' extension 
           # will be appended to create the filename for a 
           # climatological forecast that corresponds to the forecast 
           # file's simulation. This is used to calculate ACC, if the 
           # constructed filename does not exists, a climo forecast 
           # will be calculated and stored in that location. 
      }

    - eval_vairables (list): a list of strings identifying the variables over
      which RMSE and ACC will be calculated. The variable name must match the 
      variable name in the forecast file. Example list: 
      
      ['z500','t850','t2m0',]
    
    KWARGS:
    - plot (bool): Default is True. Whether to plot metrics. Determines 
      tpye of return values.
    - plot_file (str): abosolute path to save output plot.     
    - xticks (list): a list of value at which to label the xaxis

    RETURNS:
     This depends on value of plot parameter. If plot is false returns a 
     dictionary containing the metrics sorted by each forecast and 
     variable. The metrics dictionaries will be stored as numpy arrays.
     If plot is true, returns the axis and figure objects of the 
     generated plot and the metrics dictionary.
    """
 
    metrics = {}

    for eval_variable in eval_variables:
        
        for forecast in forecasts: 
            
            # create dictionary enteries in necessary                        
            if forecast['label'] not in list(metrics.keys()):
                metrics[forecast['label']] = {}
            if eval_variable not in list(metrics[forecast['label']].keys()):
                metrics[forecast['label']][eval_variable] = {}
            else:
                raise ValueError('Attemping to double calculate metrics for variable-label paire. \
Check eval_variables or forecast labels for duplicates')

            # initialize evaluator object, generate verification 
            # generate corresponding verification forecast and 
            # climatology
            evaluator = evaluators.EvaluatorHPX(
                forecast_path=forecast['filename'],
                eval_variable=eval_variable, 
                verification_path=forecast['ground_truth'][eval_variable],
                **forecast['evaluator_kwargs'])
            evaluator.generate_verification(
                verification_path=forecast['ground_truth'][eval_variable],)
            # if climatological forecast associated with file doesn't
            # exist, we need to create one
            if not os.path.isfile(f"{forecast['climatology_forecast_prefix']}{eval_variable}.nc"):
                evaluator.generate_climatology(
                    verification_path=forecast['ground_truth'][eval_variable],
                    netcdf_dst_path=f"{forecast['climatology_forecast_prefix']}{eval_variable}.nc")

            # calculate metrics using built in evaluator methods  
            metrics[forecast['label']][eval_variable]['rmse']=evaluator.get_rmse()
            metrics[forecast['label']][eval_variable]['acc']=evaluator.get_acc(
                climatology_path=f"{forecast['climatology_forecast_prefix']}{eval_variable}.nc")
            if 'fhours' not in list(metrics[forecast['label']].keys()):
                metrics[forecast['label']]['fhours']=evaluator.get_forecast_hours()

    if plot:
 
        # initialize figure and axes to accomodate evaluation variables
        fig, axs = plt.subplots(
            len(eval_variables),2,
            figsize=(10,len(eval_variables)*3),
            squeeze=False)

        for i, eval_variable in enumerate(eval_variables):
            for forecast in list(metrics.keys()):
                axs[i,0].plot(
                    metrics[forecast]['fhours'],
                    metrics[forecast][eval_variable]['rmse'],
                    label = forecast if i==0 else None,)
                if eval_variable=='t2m0': # t2m ACC are calculated as daily averages 
                    axs[i,1].plot(
                        np.arange(fhours[0],fhours[-1]+1,24),
                        metrics[forecast][eval_variable]['acc'],)
                else:
                    axs[i,1].plot(
                        metrics[forecast]['fhours'],
                        metrics[forecast][eval_variable]['acc'],)
            axs[i,0].grid()
            axs[i,1].grid()
            axs[i,0].set_xticks(xticks) 
            axs[i,1].set_xticks(xticks)
            axs[i,0].set_xticklabels([])
            axs[i,1].set_xticklabels([]) 
            axs[i,0].set_xlim([xticks[0],xticks[-1]]) 
            axs[i,1].set_xlim([xticks[0],xticks[-1]])
            axs[i,0].set_ylabel(f'{vl[eval_variable]} RMSE') 
            axs[i,1].set_ylabel(f'{vl[eval_variable]} ACC') 

        axs[i,0].set_xticklabels((axs[i,0].get_xticks()/24).astype(int))
        axs[i,1].set_xticklabels((axs[i,1].get_xticks()/24).astype(int)) 
        axs[i,0].set_xlabel('forecast day') 
        axs[i,1].set_xlabel('forecast day')

        # save plot 
        fig.tight_layout()
        fig.savefig(plot_file,dpi=200)
        return fig, axs, metrics
    else:
        return metrics    
      

