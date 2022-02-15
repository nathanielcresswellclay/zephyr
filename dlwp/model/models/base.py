from abc import abstractmethod, ABC
from typing import Optional, Sequence, Tuple, Union

from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch


class BaseModel(pl.LightningModule, ABC):
    """
    Base class with some common methods for all DELUSioN models.
    """
    def __init__(
            self,
            loss: DictConfig,
            batch_size: Optional[int] = None
    ):
        super(BaseModel, self).__init__()
        self.loss_cfg = loss
        self.metrics = None
        self.loss = None
        self.batch_size = batch_size

    @abstractmethod
    def _compute_input_channels(self) -> int:
        pass

    @abstractmethod
    def _compute_output_channels(self) -> int:
        pass

    def configure_metrics(self):
        """
        Build the metrics dictionary.
        """
        metrics = {
            'loss': instantiate(self.loss_cfg),
            'mse': torch.nn.MSELoss(),
            'mae': torch.nn.L1Loss()
        }
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
        self.log('loss', loss)

        for m, metric in self.metrics.items():
            self.log(f'val_{m}', metric(outputs, targets), prog_bar=True, sync_dist=True, batch_size=self.batch_size)

        return loss
