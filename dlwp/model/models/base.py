from abc import ABC
from typing import Optional, Sequence, Tuple, Union

from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch

from dlwp.model.losses import LossOnStep


class BaseIterativeModel(pl.LightningModule, ABC):
    """
    Base class with some common methods for all DELUSioN models.
    """
    def __init__(
            self,
            optimizer: DictConfig,
            scheduler: Optional[DictConfig],
            loss: DictConfig,
            input_channels: int,
            output_channels: int,
            n_constants: int,
            decoder_input_channels: int,
            input_time_dim: int,
            output_time_dim: int,
            cube_dim: int = 64,
            batch_size: Optional[int] = None
    ):
        super(BaseIterativeModel, self).__init__()
        self.optimizer_cfg = optimizer
        self.scheduler_cfg = scheduler
        self.loss_cfg = loss
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_constants = n_constants
        self.decoder_input_channels = decoder_input_channels
        self.input_time_dim = input_time_dim
        self.output_time_dim = output_time_dim
        self.batch_size = batch_size

        # Number of passes through the model, or a diagnostic model with only one output time
        self.is_diagnostic = self.output_time_dim == 1 and self.input_time_dim > 1
        if not self.is_diagnostic and (self.output_time_dim % self.input_time_dim != 0):
            raise ValueError(f"'output_time_dim' must be a multiple of 'input_time_dim' (got "
                             f"{self.output_time_dim} and {self.input_time_dim})")

        # Example input arrays. Expects from data loader a sequence of (inputs, [decoder_inputs, [constants]])
        # inputs: [B, input_time_dim, input_channels, F, H, W]
        # decoder_inputs: [B, input_time_dim + output_time_dim, decoder_input_channels, F, H, W]
        # constants: [n_constants, F, H, W]
        example_input_array = [torch.randn(1, self.input_time_dim, self.input_channels, 6, cube_dim, cube_dim)]
        if self.decoder_input_channels > 0:
            example_input_array.append(
                torch.randn(1, self.input_time_dim + self.output_time_dim, self.decoder_input_channels,
                            6, cube_dim, cube_dim)
            )
        if self.n_constants > 0:
            example_input_array.append(
                torch.randn(self.n_constants, 6, cube_dim, cube_dim)
            )
        # Wrap in list so that Lightning interprets it correctly as a single arg to `forward`
        self.example_input_array = [example_input_array]

        self.loss = None
        self.metrics = None
        self.configure_metrics()

    def _compute_input_channels(self) -> int:
        return self.input_time_dim * (self.input_channels + self.decoder_input_channels) + self.n_constants

    def _compute_output_channels(self) -> int:
        return (1 if self.is_diagnostic else self.input_time_dim) * self.output_channels

    def _reshape_inputs(self, inputs: Sequence, step: int = 0) -> torch.Tensor:
        """
        Returns a single tensor to pass into the model encoder/decoder. Squashes the time/channel dimension and
        concatenates in constants and decoder inputs.
        :param inputs: list of expected input tensors (inputs, decoder_inputs, constants)
        :param step: step number in the sequence of integration_steps
        :return: reshaped Tensor in expected shape for model encoder
        """
        if not (self.n_constants > 0 or self.decoder_input_channels > 0):
            return inputs[0].flatten(start_dim=1, end_dim=2)
        elif self.n_constants == 0:
            result = [
                inputs[0].flatten(start_dim=1, end_dim=2),  # inputs
                inputs[1][:, slice(step * self.input_time_dim, (step + 1) * self.input_time_dim)].flatten(1, 2)  # DI
            ]
            return torch.cat(result, dim=1)
        elif self.decoder_input_channels == 0:
            result = [
                inputs[0].flatten(start_dim=1, end_dim=2),  # inputs
                inputs[2].expand(*tuple([inputs[0].shape[0]] + len(inputs[2].shape) * [-1]))  # constants
            ]
            return torch.cat(result, dim=1)
        else:
            result = [
                inputs[0].flatten(start_dim=1, end_dim=2),  # inputs
                inputs[1][:, slice(step * self.input_time_dim, (step + 1) * self.input_time_dim)].flatten(1, 2),  # DI
                inputs[2].expand(*tuple([inputs[0].shape[0]] + len(inputs[2].shape) * [-1]))  # constants
            ]
            return torch.cat(result, dim=1)

    def _reshape_outputs(self, outputs: torch.Tensor) -> torch.Tensor:
        shape = tuple(outputs.shape)
        return outputs.view(shape[0], 1 if self.is_diagnostic else self.input_time_dim, -1, *shape[2:])

    @property
    def integration_steps(self):
        return max(self.output_time_dim // self.input_time_dim, 1)

    def configure_metrics(self):
        """
        Build the metrics dictionary.
        """
        metrics = {
            'loss': instantiate(self.loss_cfg),
            'mse': torch.nn.MSELoss(),
            'mae': torch.nn.L1Loss()
        }
        for step in range(self.integration_steps):
            metrics[f'loss_{step}'] = LossOnStep(metrics['loss'], self.input_time_dim, step)
        self.metrics = torch.nn.ModuleDict(metrics)
        self.loss = self.metrics['loss']

    def training_step(
            self,
            batch: Tuple[Union[Sequence, torch.Tensor], torch.Tensor],
            batch_idx: int
    ) -> torch.Tensor:
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss(outputs, targets)
        self.log('loss', loss, batch_size=self.batch_size)
        return loss

    def validation_step(
            self,
            batch: Tuple[Union[Sequence, torch.Tensor], torch.Tensor],
            batch_idx: int
    ) -> torch.Tensor:
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss(outputs, targets)
        self.log('loss', loss, sync_dist=True, batch_size=self.batch_size)

        for m, metric in self.metrics.items():
            self.log(f'val_{m}', metric(outputs, targets), prog_bar=True, sync_dist=True, batch_size=self.batch_size)

        return loss
