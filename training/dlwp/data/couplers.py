import xarray as xr 
import pandas as pd  
from typing import DefaultDict, Optional, Sequence, Union

class GroundTruthCoupler():

    def __init__(
        self,
        dataset: xr.Dataset,
        variables: Sequence = [],
        method: str = 'update',
        input_times: Sequence = [pd.Timedelta('48h'), pd.Timedelta('96H')],
    ):
        """
        coupler used to inferface two components of the earch system
        """
        self.ds = dataset
        self.variables = variables 
        self.method = method
        self.input_times = input_times 
        
        
