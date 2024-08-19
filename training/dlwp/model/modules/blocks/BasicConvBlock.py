from typing import Any, Dict, Optional, Sequence, Union

from hydra.utils import instantiate
from omegaconf import DictConfig
import torch as th

from training.dlwp.model.modules.healpix import HEALPixLayer

#
# Basic Conv block
#

class BasicConvBlock(th.nn.Module):
    """
    Convolution block consisting of n subsequent convolutions and activations
    """
    def __init__(
            self,
            geometry_layer: th.nn.Module = HEALPixLayer,
            in_channels: int = 3,
            out_channels: int = 1,
            kernel_size: int = 3,
            dilation: int = 1,
            n_layers: int = 1,
            latent_channels: int = None,
            activation: th.nn.Module = None,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False
            ):
        super().__init__()
        if latent_channels is None: latent_channels = max(in_channels, out_channels)
        convblock = []
        for n in range(n_layers):
            convblock.append(geometry_layer(
                layer='torch.nn.Conv2d',
                in_channels=in_channels if n == 0 else latent_channels,
                out_channels=out_channels if n == n_layers - 1 else latent_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad
                ))
            if activation is not None: convblock.append(activation)
        self.convblock = th.nn.Sequential(*convblock)

    def forward(self, x):
        return self.convblock(x)

