from typing import Any, Dict, Optional, Sequence, Union

from hydra.utils import instantiate
from omegaconf import DictConfig
import torch as th

from training.dlwp.model.modules.healpix import HEALPixLayer
from training.dlwp.model.modules.utils import Interpolate

#
# UPSAMPLING BLOCKS
#

class InterpolationUpsample(th.nn.Module):
    def __init__(
            self,
            geometry_layer: th.nn.Module = HEALPixLayer,
            in_channels: int = 3,
            out_channels: int = 1,
            kernel_size: int = 3,
            mode: str = "nearest",
            upsampling: int = 2,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False
            ):
        super().__init__()
        self.upsampler = geometry_layer(
            layer=Interpolate,
            scale_factor=upsampling,
            mode=mode,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            )
    def forward(self, x):
        return self.upsampler(x)


class TransposedConvUpsample(th.nn.Module):
    def __init__(
            self,
            geometry_layer: th.nn.Module = HEALPixLayer,
            in_channels: int = 3,
            out_channels: int = 1,
            upsampling: int = 2,
            activation: th.nn.Module = None,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False
            ):
        super().__init__()
        upsampler = []
        # Upsample transpose conv
        upsampler.append(geometry_layer(
            layer='torch.nn.ConvTranspose2d',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=upsampling,
            stride=upsampling,
            padding=0,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            ))
        if activation is not None:
            upsampler.append(activation)
        self.upsampler = th.nn.Sequential(*upsampler)

    def forward(self, x):
        return self.upsampler(x)
