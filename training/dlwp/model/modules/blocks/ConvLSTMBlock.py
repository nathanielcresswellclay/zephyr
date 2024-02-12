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
            geometry_layer: nn.Module = HEALPixLayer,
            in_channels: int = 3,
            out_channels: int = 1,
            latent_channels: int = 1, # or 3
            kernel_size: int = 3, # encoding convolution size
            downscale_factor: int = 4,
            upscale_factor: int = 4,
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
        # Linear postprocessing -- was the previous way to get 3x3, but now we should use the LSTM bit
        # convblock.append(geometry_layer(
        #     layer='torch.nn.Conv2d',
        #     in_channels=int(latent_channels*upscale_factor),
        #     out_channels=out_channels,
        #     kernel_size=latent_conv_size,
        #     enable_nhwc=enable_nhwc,
        #     enable_healpixpad=enable_healpixpad
        #     ))
        # self.convblock = th.nn.Sequential(*convblock)

        # Now for the LSTM bit
        self.conv_gates = geometry_layer(
            layer="torch.nn.Conv2d",
            in_channels=in_channels + self.channels,
            out_channels=2*self.channels,  # for update_gate,reset_gate respectively
            kernel_size=latent_conv_size,
            padding="same",
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            )
        self.conv_can = geometry_layer(
            layer="torch.nn.Conv2d",
            in_channels=in_channels+self.channels,
            out_channels=self.channels, # for candidate neural memory
            kernel_size=kernel_size,
            padding="same",
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad
            )

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
        self.h = torch.zeros(1, 1, 1, 1)
        self.c = torch.zeros(1, 1, 1, 1)  # LSTM has 4 states


    def forward(self, inputs: Sequence) -> Sequence:
        if inputs.shape != self.h.shape:
            self.h = torch.zeros_like(inputs)
            self.c = torch.zeros_like(inputs)

        combined = torch.cat([inputs, self.h], dim=1)
        combined_conv = self.conv_gates(combined)

        # Split the combined_conv into input_gate, forget_gate, cell_gate, output_gate
        i, f, c_hat, o = torch.split(combined_conv, self.channels, dim=1)

        input_gate = torch.sigmoid(i)
        forget_gate = torch.sigmoid(f)
        cell_gate = torch.tanh(c_hat)
        output_gate = torch.sigmoid(o)

        self.c = forget_gate * self.c + input_gate * cell_gate
        self.h = output_gate * torch.tanh(self.c)

        return inputs + self.h

        return self.skip_module(x) + self.convblock(x)


    def reset(self):
        self.h = torch.zeros_like(self.h)
        self.c = torch.zeros_like(self.c)
