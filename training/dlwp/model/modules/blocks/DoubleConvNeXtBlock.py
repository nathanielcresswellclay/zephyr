from typing import Any, Dict, Optional, Sequence, Union

from hydra.utils import instantiate
from omegaconf import DictConfig
import torch as th

from training.dlwp.model.modules.healpix import HEALPixLayer

class DoubleConvNeXtBlock(th.nn.Module):
    """
    Another modification of ConvNeXtBlock block this time putting two into a single block
    """
    def __init__(
            self,
            geometry_layer: th.nn.Module = HEALPixLayer,
            in_channels: int = 3,
            latent_channels: int = 1,
            out_channels: int = 1,
            kernel_size: int = 3,
            dilation: int = 1,
            upscale_factor: int = 4,
            n_layers: int = 1,
            activation: th.nn.Module = None,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False
            ):
        super().__init__()
    
        if in_channels == int(latent_channels):
            self.skip_module1 = lambda x: x  # Identity-function required in forward pass
        else:
            self.skip_module1 = geometry_layer(
                layer='torch.nn.Conv2d',
                in_channels=in_channels,
                out_channels=int(latent_channels),
                kernel_size=1,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad
                )
        if out_channels == int(latent_channels):
            self.skip_module2 = lambda x: x  # Identity-function required in forward pass
        else:
            self.skip_module2 = geometry_layer(
                layer='torch.nn.Conv2d',
                in_channels=int(latent_channels),
                out_channels=out_channels,
                kernel_size=1,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad
                )

        # 1st ConvNeXt block, the output of this one remains internal 
        convblock1 = []
        # 3x3 convolution establishing latent channels channels
        convblock1.append(geometry_layer(
            layer='torch.nn.Conv2d',
            in_channels=in_channels,
            out_channels=int(latent_channels),
            kernel_size=kernel_size,
            dilation=dilation,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            ))
        if activation is not None: convblock1.append(activation)
        # 1x1 convolution establishing increased channels
        convblock1.append(geometry_layer(
            layer='torch.nn.Conv2d',
            in_channels=int(latent_channels),
            out_channels=int(latent_channels*upscale_factor),
            kernel_size=1,
            dilation=dilation,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            ))
        if activation is not None: convblock1.append(activation)
        # 1x1 convolution returning to latent channels
        convblock1.append(geometry_layer(
            layer='torch.nn.Conv2d',
            in_channels=int(latent_channels*upscale_factor),
            out_channels=int(latent_channels),
            kernel_size=1,
            dilation=dilation,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            ))
        if activation is not None: convblock1.append(activation)
        self.convblock1 = th.nn.Sequential(*convblock1)

        # 2nd ConNeXt block, takes the output of the first convnext block 
        convblock2 = []
        # 3x3 convolution establishing latent channels channels
        convblock2.append(geometry_layer(
            layer='torch.nn.Conv2d',
            in_channels=int(latent_channels),
            out_channels=int(latent_channels),
            kernel_size=kernel_size,
            dilation=dilation,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            ))
        if activation is not None: convblock2.append(activation)
        # 1x1 convolution establishing increased channels
        convblock2.append(geometry_layer(
            layer='torch.nn.Conv2d',
            in_channels=int(latent_channels),
            out_channels=int(latent_channels*upscale_factor),
            kernel_size=1,
            dilation=dilation,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            ))
        if activation is not None: convblock2.append(activation)
        # 1x1 convolution reducing to output channels
        convblock2.append(geometry_layer(
            layer='torch.nn.Conv2d',
            in_channels=int(latent_channels*upscale_factor),
            out_channels=out_channels,
            kernel_size=1,
            dilation=dilation,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            ))
        if activation is not None: convblock2.append(activation)
        self.convblock2 = th.nn.Sequential(*convblock2)
    


    def forward(self, x):
        # internal convnext result 
        x1 = self.skip_module1(x) + self.convblock1(x)
        # return second convnext result 
        return self.skip_module2(x1) + self.convblock2(x1)
      