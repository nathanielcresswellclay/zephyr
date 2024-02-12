from typing import Any, Dict, Optional, Sequence, Union

from hydra.utils import instantiate
from omegaconf import DictConfig
import torch as th

from training.dlwp.model.modules.healpix import HEALPixLayer

#
# One of our recurrent blocks
#

class ConvLSTMBlock(th.nn.Module):
    def __init__(
            self,
            geometry_layer: th.nn.Module = HEALPixLayer,
            in_channels: int = 3,
            out_channels: int = 1,
            latent_channels: int = 1,
            kernel_size: int = 3,
            downscale_factor: int = 4,
            upscale_factor: int = 4,
            latent_conv_size: int = 3,  # Add latent_conv_size parameter
            dilation: int = 1,
            activation: th.nn.Module = None,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False
            ):
        super().__init__()

        # Instantiate 1x1 conv to increase/decrease channel depth if necessary
        # Skip connection for output
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

        # Now for the LSTM bit
        self.channels = in_channels
        self.conv_gates = geometry_layer(
            layer="torch.nn.Conv2d",
            in_channels=in_channels + self.channels,
            out_channels=4*self.channels,  # for input_gate, forget_gate, cell_gate, output_gate respectively (LSTM)
            kernel_size=kernel_size,
            padding="same",
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            )
        self.h = th.zeros(1, 1, 1, 1)
        self.c = th.zeros(1, 1, 1, 1)  # LSTM has 4 states


    def forward(self, inputs: Sequence) -> Sequence:
        # First run our convblock step
        conv_outputs = self.convblock(inputs)

        if conv_outputs.shape != self.h.shape:
            self.h = th.zeros_like(inputs)
            self.c = th.zeros_like(inputs)

        combined = th.cat([conv_outputs, self.h], dim=1)
        combined_conv = self.conv_gates(combined)

        # Split the combined_conv into input_gate, forget_gate, cell_gate, output_gate
        i, f, c_hat, o = th.split(combined_conv, self.channels, dim=1)

        input_gate = th.sigmoid(i)
        forget_gate = th.sigmoid(f)
        cell_gate = th.tanh(c_hat)
        output_gate = th.sigmoid(o)

        self.c = forget_gate * self.c + input_gate * cell_gate
        self.h = output_gate * th.tanh(self.c)

        conv_intermediate = self.skip_module(inputs)  + self.h
        return 


    def reset(self):
        self.h = th.zeros_like(self.h)
        self.c = th.zeros_like(self.c)
