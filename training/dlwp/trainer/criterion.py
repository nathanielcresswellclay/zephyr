import torch as th
from typing import Any, Dict, Optional, Sequence, Union

"""
Custom loss classes allow for more sophisticated training optimization.

Each custom loss should inherit all methods of th.nn._Loss base class or subclasses thereof. 
Additionally, custom loss classes should define a setup function which receives the trainer object. 
The setup function should be used to move tensors to appropriate gpus and finalize configuration
of the loss calculation using information about the model (trainer.model) and trainer. Custom
losses should also redefine the forward function to contain a flag indicating whether or not to 
average output channels. This is used in the varible wise logging of validation loss by the trainer. 

"""
class BaseMSE( th.nn.MSELoss ):
    """
    Base MSE class offers impementaion for basic MSE loss compatable with dlwp custom loss training
    """
    def __init__(
        self,
    ):
        super().__init__()
        self.device = None
    def setup(self, trainer):
        """
        Nothing to implement here  
        """
        pass
    def forward(self, prediction, target, average_channels=True ):
        d = ((target-prediction)**2).mean(dim=(0, 1, 2, 4, 5))
        if average_channels: 
            return th.mean(d)
        else: 
            return d

class WeightedMSE( th.nn.MSELoss ):

    """
    Loss object that allows for user defined weighting of variables when calculating MSE
    """

    def __init__(
        self,
        weights: Sequence = [],
    ):
    
        """
        params: weights: list of floats that determine weighting of variable loss, assumed to be
            in order consistent with order of model output channels
        """
        super().__init__()
        self.loss_weights = th.tensor(weights)
        self.device = None

    def setup(self, trainer):
        """
        pushes weights to cuda device 
        """
        
        try:
            assert len(trainer.output_variables) == len(self.loss_weights)
        except AssertionError:
            raise ValueError('Length of outputs and loss_weights is not the same!')

        self.loss_weights = self.loss_weights.to(device=trainer.device)

    def forward(self, prediction, target, average_channels=True ):

        d = ((target-prediction)**2).mean(dim=(0, 1, 2, 4, 5))*self.loss_weights
        if average_channels: 
            return th.mean(d)
        else: 
            return d
