from typing import Any, Dict, Optional, Sequence, Union

from hydra.utils import instantiate
from omegaconf import DictConfig
import torch as th

from training.dlwp.model.modules.healpix import HEALPixLayer

#
# FOLDING/UNFOLDING BLOCKS
#

class FoldFaces(th.nn.Module):
    # perform face folding:
    # [B, F, C, H, W] -> [B*F, C, H, W]

    def __init__(self):
        super().__init__()

    def forward(self, tensor: th.Tensor) -> th.Tensor:

        N, F, C, H, W = tensor.shape
        tensor = th.reshape(tensor, shape=(N*F, C, H, W))
    
        return tensor


class UnfoldFaces(th.nn.Module):
    # perform face unfolding:
    # [B*F, C, H, W] -> [B, F, C, H, W]

    def __init__(self, num_faces=12):
        super().__init__()
        self.num_faces = num_faces

    def forward(self, tensor: th.Tensor) -> th.Tensor:
        
        NF, C, H, W = tensor.shape
        tensor = th.reshape(tensor, shape=(-1, self.num_faces, C, H, W))
    
        return tensor
