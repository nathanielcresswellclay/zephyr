from typing import Any, Dict, Optional, Sequence, Union

from hydra.utils import instantiate
from omegaconf import DictConfig
import torch as th

from training.dlwp.model.modules.healpix import HEALPixLayer

class ConvNeXtBlock(th.nn.Module):
    """
    A modification of the convolution block reported in Figure 4 of https://arxiv.org/pdf/2201.03545.pdf
    """
    def __init__(
            self,
            geometry_layer: th.nn.Module = HEALPixLayer,
            in_channels: int = 3,
            latent_channels: int = 1,
            out_channels: int = 1,
            kernel_size: int = 3, # encoding convolution size
            latent_conv_size: int = 1, # decoding convolution size. if 0, do not include this layer
            dilation: int = 1,
            upscale_factor: int = 4,
            n_layers: int = 1,
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
        # 3x3 convolution increasing channels
        convblock.append(geometry_layer(
            layer='torch.nn.Conv2d',
            in_channels=in_channels,
            out_channels=int(latent_channels*upscale_factor),
            kernel_size=kernel_size,
            dilation=dilation,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            ))
        if activation is not None: convblock.append(activation)
        # 3x3 convolution maintaining increased channels
        convblock.append(geometry_layer(
            layer='torch.nn.Conv2d',
            in_channels=int(latent_channels*upscale_factor),
            out_channels=int(latent_channels*upscale_factor),
            kernel_size=kernel_size,
            dilation=dilation,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            ))
        if activation is not None: convblock.append(activation)
        # Linear postprocessing. Do not do this when pairing with a downstream block that has its own convolution gates
        if latent_conv_size is not 0:
            convblock.append(geometry_layer(
                layer='torch.nn.Conv2d',
                in_channels=int(latent_channels*upscale_factor),
                out_channels=out_channels,
                kernel_size=latent_conv_size,
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad
                ))
        self.convblock = th.nn.Sequential(*convblock)

    def forward(self, x):
        return self.skip_module(x) + self.convblock(x)
