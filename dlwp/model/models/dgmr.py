from abc import ABC
import logging
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
from torch.nn.utils.parametrizations import spectral_norm

from dlwp.model.losses import loss_hinge_disc, loss_hinge_gen, LossOnStep
from dlwp.model.models.base import BaseModel
from dlwp.model.layers.cube_sphere import CubeSpherePadding, CubeSphereLayer
from dlwp.model.models.unet import IterativeUnet

logger = logging.getLogger(__name__)


class CubeSphereUnetDGMR(BaseModel, ABC):
    def __init__(
            self,
            encoder: DictConfig,
            decoder: DictConfig,
            discriminator: DictConfig,
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
            batch_size: Optional[int] = None,
            disc_steps_per_iter: int = 2
    ):
        """
        Pytorch-lightning module implementation of the Deep Learning Weather Prediction (DLWP) U-net model on the
        cube sphere grid, with a GAN discriminator based on the Deep Generative Model for Radar:
            https://www.nature.com/articles/s41586-021-03854-z

        :param encoder: dictionary of instantiable parameters for the U-net encoder (see UnetEncoder docs)
        :param decoder: dictionary of instantiable parameters for the U-net decoder (see UnetDecoder docs)
        :param discriminator: dictionary of instantiable parameters for the DGMR discriminator (see DgmrDiscriminator)
        :param optimizer: dictionary of instantiable parameters for the model optimizers. Contains two keys,
            "generator" and "discriminator", for the generator and discriminator components.
        :param scheduler: dictionary of instantiable parameters for optimizer scheduler
        :param loss: dictionary of instantiable parameters for the model loss function
        :param input_channels: number of input channels expected in the input array schema. Note this should be the
            number of input variables in the data, NOT including data reshaping for the encoder part.
        :param output_channels: number of output channels expected in the output array schema, or output variables
        :param n_constants: number of optional constants expected in the input arrays. If this is zero, no constants
            should be provided as inputs to `forward`.
        :param decoder_input_channels: number of optional prescribed variables expected in the decoder input array
            for both inputs and outputs. If this is zero, no decoder inputs should be provided as inputs to `forward`.
        :param input_time_dim: number of time steps in the input array
        :param output_time_dim: number of time steps in the output array
        :param cube_dim: number of points on the side of a cube face
        :param batch_size: batch size. Provided only to correctly compute validation losses in metrics.
        """
        super().__init__(loss, batch_size=batch_size)
        self.optimizer_cfg = optimizer
        self.scheduler_cfg = scheduler
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_constants = n_constants
        self.decoder_input_channels = decoder_input_channels
        self.input_time_dim = input_time_dim
        self.output_time_dim = output_time_dim
        self.disc_steps_per_iter = disc_steps_per_iter

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

        # Build the generator and discriminator
        self.generator = IterativeUnet(
            encoder=encoder,
            decoder=decoder,
            input_channels=self.input_channels,
            output_channels=self.output_channels,
            n_constants=self.n_constants,
            decoder_input_channels=self.decoder_input_channels,
            input_time_dim=self.input_time_dim,
            output_time_dim=self.output_time_dim,
        )
        self.discriminator = instantiate(discriminator, input_channels=self.input_channels)

        self.save_hyperparameters()
        self.configure_metrics()

        self.global_iteration = 0

        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        torch.autograd.set_detect_anomaly(True)

    def configure_metrics(self):
        """
        Build the metrics dictionary. Handled differently because some metrics require disc scores.
        """
        metrics = {
            'mse': torch.nn.MSELoss(),
            'mae': torch.nn.L1Loss()
        }
        for step in range(self.generator.integration_steps):
            metrics[f'mse_{step}'] = LossOnStep(metrics['mse'], self.input_time_dim, step)
        self.metrics = torch.nn.ModuleDict(metrics)
        self.loss = instantiate(self.loss_cfg)

    def configure_optimizers(
            self
    ) -> Union[torch.optim.Optimizer, Dict[str, Any]]:
        g_optimizer = instantiate(self.optimizer_cfg.generator, self.generator.parameters())
        d_optimizer = instantiate(self.optimizer_cfg.discriminator, self.discriminator.parameters())
        if self.scheduler_cfg is not None:
            scheduler = instantiate(self.scheduler_cfg, optimizer=g_optimizer)
            g_optimizer = {
                'optimizer': g_optimizer,
                'lr_scheduler': scheduler
            }
            d_optimizer = {
                'optimizer': d_optimizer
            }
        return [g_optimizer, d_optimizer]

    def forward(self, inputs: Sequence, output_only_last=False) -> torch.Tensor:
        return self.generator(inputs, output_only_last)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        batch_size = inputs[0].shape[0]

        self.global_iteration += 1
        g_opt, d_opt = self.optimizers()
        scheduler = self.lr_schedulers()

        # Multiple discriminator steps per generator step
        for _ in range(self.disc_steps_per_iter):
            # Make a prediction with the generator. Must make at each step because tensors are used in backward().
            prediction = self(inputs)
            # Cat along time dimension [B, I+O, C, H, W]
            generated_sequence = torch.cat([inputs[0], prediction], dim=1)
            real_sequence = torch.cat([inputs[0], targets], dim=1)
            # Concatenate fake and real sequences in batch for processing
            concatenated_inputs = torch.cat([real_sequence, generated_sequence], dim=0)

            concatenated_outputs = self.discriminator(concatenated_inputs)
            score_real, score_generated = torch.split(concatenated_outputs, batch_size, dim=0)
            discriminator_loss = loss_hinge_disc(score_generated, score_real)
            d_opt.zero_grad()
            self.manual_backward(discriminator_loss)
            d_opt.step()

        # Make a prediction with the generator
        prediction = self(inputs)
        # Cat along time dimension [B, I+O, C, H, W]
        generated_sequence = torch.cat([inputs[0], prediction], dim=1)

        # Update generator with latest discriminator prediction
        score_generated = self.discriminator(generated_sequence)
        generator_disc_loss = loss_hinge_gen(score_generated)
        generator_loss = self.loss(prediction, targets, generator_disc_loss)
        g_opt.zero_grad()
        self.manual_backward(generator_loss)
        g_opt.step()

        # Step the scheduler
        scheduler.step(generator_loss)

        self.log_dict(
            {
                "train/d_loss": discriminator_loss,
                "train/g_loss": generator_loss,
            },
            prog_bar=True,
        )

    def validation_step(
            self,
            batch: Tuple[Union[Sequence, torch.Tensor], torch.Tensor],
            batch_idx: int
    ) -> torch.Tensor:
        inputs, targets = batch
        outputs = self(inputs)
        output_sequence = torch.cat([inputs[0], outputs], dim=1)
        disc_score = loss_hinge_gen(self.discriminator(output_sequence))
        loss = self.loss(outputs, targets, disc_score)
        self.log('val_loss', loss, sync_dist=True, batch_size=self.batch_size)

        for m, metric in self.metrics.items():
            self.log(f'val_{m}', metric(outputs, targets), prog_bar=False, sync_dist=True, batch_size=self.batch_size)

        return loss


class DBlock(torch.nn.Module):
    def __init__(
            self,
            input_channels: int,
            output_channels: int,
            conv_type: str = "Conv2d",
            first_relu: bool = True,
            keep_same_output: bool = False,
            keep_time_dim: bool = False
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.first_relu = first_relu
        self.keep_same_output = keep_same_output
        self.keep_time_dim = keep_time_dim

        assert conv_type in ['Conv2d', 'Conv3d']
        self.conv_type = conv_type
        pool_config = DictConfig(dict(
            _target_=f"torch.nn.{'AvgPool3d' if self.conv_type == 'Conv3d' else 'AvgPool2d'}",
            kernel_size=(1, 2, 2) if (self.conv_type == 'Conv3d' and self.keep_time_dim) else 2
        ))
        self.pooling = CubeSphereLayer(pool_config, add_polar_layer=False, flip_north_pole=False)
        conv_1x1_config = DictConfig(dict(
            _target_=f"torch.nn.{self.conv_type}",
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=1
        ))
        self.conv_1x1 = CubeSphereLayer(conv_1x1_config, use_spectral_norm=True)
        conv_3x3_config = DictConfig(dict(
            _target_=f"torch.nn.{self.conv_type}",
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=3,
            padding=(1, 0, 0) if self.conv_type == 'Conv3d' else 0
        ))
        self.first_conv_3x3 = torch.nn.Sequential(
            CubeSpherePadding(),
            CubeSphereLayer(conv_3x3_config, use_spectral_norm=True)
        )
        conv_3x3_config['in_channels'] = output_channels
        self.last_conv_3x3 = torch.nn.Sequential(
            CubeSpherePadding(),
            CubeSphereLayer(conv_3x3_config, use_spectral_norm=True)
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_channels != self.output_channels:
            x1 = self.conv_1x1(x)
            if not self.keep_same_output:
                x1 = self.pooling(x1)
        else:
            x1 = x

        if self.first_relu:
            x = self.relu(x)
        x = self.first_conv_3x3(x)
        x = self.relu(x)
        x = self.last_conv_3x3(x)

        if not self.keep_same_output:
            x = self.pooling(x)
        x = x1 + x  # Sum the outputs should be half spatial and double channels
        return x


class TemporalDiscriminator(torch.nn.Module):
    def __init__(
            self,
            input_channels: int,
            num_layers: int = 3,
            internal_channels: int = 8,
            keep_time_dim: bool = False
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_layers = num_layers
        self.internal_channels = internal_channels
        self.keep_time_dim = keep_time_dim

        # self.downsample = torch.nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.d1 = DBlock(
            input_channels=self.input_channels,
            output_channels=self.internal_channels * self.input_channels,
            conv_type='Conv3d',
            first_relu=False,
            keep_time_dim=self.keep_time_dim
        )
        self.d2 = DBlock(
            input_channels=self.internal_channels * self.input_channels,
            output_channels=2 * self.internal_channels * self.input_channels,
            conv_type='Conv3d',
            keep_time_dim=self.keep_time_dim
        )
        self.intermediate_dblocks = torch.nn.ModuleList()
        new_int_channels = int(self.internal_channels)
        for _ in range(num_layers):
            new_int_channels *= 2
            self.intermediate_dblocks.append(
                DBlock(
                    input_channels=new_int_channels * self.input_channels,
                    output_channels=2 * new_int_channels * self.input_channels,
                    conv_type='Conv2d',
                )
            )

        self.d_last = DBlock(
            input_channels=2 * new_int_channels * self.input_channels,
            output_channels=2 * new_int_channels * self.input_channels,
            keep_same_output=True,
            conv_type='Conv2d',
        )

        self.fc = spectral_norm(torch.nn.Linear(2 * new_int_channels * self.input_channels, 1))
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(2 * new_int_channels * self.input_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Permute time and channels to B, C, T, F, H, W
        x = torch.permute(x, dims=(0, 2, 1, 3, 4, 5))
        # 2 residual 3D blocks to halve resolution if image, double number of channels and reduce
        # number of time steps
        x = self.d1(x)
        x = self.d2(x)
        # Convert back to B, T, C, F, H, W
        x = torch.permute(x, dims=(0, 2, 1, 3, 4, 5))
        # Compute the 2D D-blocks on each time step individually
        representations = []
        for time_step in range(x.size(1)):
            # Intermediate DBlocks
            # Three residual D Blocks to halve the resolution of the image and double
            # the number of channels.
            rep = x[:, time_step]
            for d in self.intermediate_dblocks:
                rep = d(rep)
            # One more D Block without downsampling or increase number of channels
            rep = self.d_last(rep)
            # Sum over the remaining spatial dims
            rep = torch.sum(self.relu(rep), dim=[3, 4])
            # Average over cube faces
            rep = torch.mean(rep, dim=-1)
            rep = self.bn(rep)
            rep = self.fc(rep)

            representations.append(rep)
        # The representations are summed together before the ReLU
        x = torch.stack(representations, dim=1)
        # Should be [Batch, N, 1]
        x = torch.sum(x, dim=1)
        return x
