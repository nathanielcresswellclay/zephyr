from typing import Any, Dict, Optional, Sequence, Union

from hydra.utils import instantiate
from omegaconf import DictConfig
import torch as th

from training.dlwp.model.modules.healpix import HEALPixLayer

#
# One of our recurrent blocks
#

class ConvGRUBlock(th.nn.Module):
    """
    Code modified from
    https://github.com/happyjin/ConvGRU-pytorch/blob/master/convGRU.py
    """
    def __init__(
            self,
            geometry_layer: th.nn.Module = HEALPixLayer,
            in_channels: int = 3,
            kernel_size: int = 1,
            downscale_factor: int = 4,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False
            ):
        super().__init__()

        self.channels = in_channels
        self.conv_gates = geometry_layer(
            layer="torch.nn.Conv2d",
            in_channels=in_channels + self.channels,
            out_channels=2*self.channels,  # for update_gate,reset_gate respectively
            kernel_size=kernel_size,
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
        self.h = th.zeros(1, 1, 1, 1)

    def forward(self, inputs: Sequence) -> Sequence:
        if inputs.shape != self.h.shape:
            self.h = th.zeros_like(inputs)
        combined = th.cat([inputs, self.h], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = th.split(combined_conv, self.channels, dim=1)
        reset_gate = th.sigmoid(gamma)
        update_gate = th.sigmoid(beta)

        combined = th.cat([inputs, reset_gate*self.h], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = th.tanh(cc_cnm)

        h_next = (1 - update_gate) * self.h + update_gate * cnm
        self.h = h_next

        return inputs + h_next

    def reset(self):
        self.h = th.zeros_like(self.h)
