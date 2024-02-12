from typing import Any, Dict, Optional, Sequence, Union

from hydra.utils import instantiate
from omegaconf import DictConfig
import torch as th

from training.dlwp.model.modules.healpix import HEALPixLayer

#
# DOWNSAMPLING BLOCKS
#

class MaxPool(th.nn.Module):
    def __init__(
            self,
            geometry_layer: th.nn.Module = HEALPixLayer,
            pooling: int = 2,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False
            ):
        super().__init__()
        self.maxpool = geometry_layer(
            layer="torch.nn.MaxPool2d",
            kernel_size=pooling,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            )
    def forward(self, x):
        return self.maxpool(x)


class AvgPool(th.nn.Module):
    def __init__(
            self,
            geometry_layer: th.nn.Module = HEALPixLayer,
            pooling: int = 2,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False
            ):
        super().__init__()
        self.avgpool = geometry_layer(
            layer="torch.nn.AvgPool2d",
            kernel_size=pooling,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            )
    def forward(self, x):
        return self.avgpool(x)


class LearnedPool(th.nn.Module):
    def __init__(
            self,
            geometry_layer: th.nn.Module = HEALPixLayer,
            in_channels: int = 1,
            out_channels: int = 1,
            pooling: int = 2,
            activation: th.nn.Module = None,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False
            ):
        super().__init__()
        # "Skip" connection
        self.skip_pool = MaxPool(
            geometry_layer=geometry_layer,
            pooling=pooling,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            )
        # Donwpooling convolution
        downpooler = []
        downpooler.append(geometry_layer(
            layer='torch.nn.Conv2d',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=pooling,
            stride=pooling,
            padding=0,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            ))
        if activation is not None:
            downpooler.append(activation)
        self.downpooler = th.nn.Sequential(*downpooler)

    def forward(self, x):
        return self.skip_pool(x) + self.downpooler(x)

