from typing import Any, Dict, Optional, Sequence, Union

from hydra.utils import instantiate
from omegaconf import DictConfig
import torch as th

from training.dlwp.model.modules.healpix import HEALPixLayer

class MobileNetConvBlock(th.nn.Module):
    """
    A convolution block as reported in Figure 4 (d) of https://arxiv.org/pdf/1801.04381.pdf

    Does not seem to improve performance over two simple convolutions
    """
    def __init__(
            self,
            geometry_layer: th.nn.Module = HEALPixLayer,
            in_channels: int = 3,
            out_channels: int = 1,
            kernel_size: int = 3,
            dilation: int = 1,
            activation: th.nn.Module = None,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False
            ):
        super().__init__()
        # Instantiate 1x1 conv to increase/decrease channel depth if necessary
        if in_channels == out_channels:
            self.skip_module = lambda x: x  # Identity-function required in forward pass
        else:
            self.skip_module = geometry_layer(
                layer='torch.nn.Conv2d',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad
                )
        # Convolution block
        convblock = []
        # Map channels to output depth
        convblock.append(geometry_layer(
            layer='torch.nn.Conv2d',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            ))
        if activation is not None: convblock.append(activation)
        # Depthwise convolution
        convblock.append(geometry_layer(
            layer='torch.nn.Conv2d',
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=out_channels,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            ))
        if activation is not None: convblock.append(activation)
        # Linear postprocessing
        convblock.append(geometry_layer(
            layer='torch.nn.Conv2d',
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            ))
        self.convblock = th.nn.Sequential(*convblock)

    def forward(self, x):
        return self.skip_module(x) + self.convblock(x)