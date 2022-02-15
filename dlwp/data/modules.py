# System modules
import logging
from abc import ABC
from typing import Optional, Union, Sequence, DefaultDict

# External modules
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# Internal modules
from .data_loading import open_time_series_dataset_classic, TimeSeriesDataset

logger = logging.getLogger(__name__)


class TimeSeriesDataModule(pl.LightningDataModule, ABC):
    def __init__(
            self,
            directory: str = '.',
            prefix: Union[None, str] = None,
            suffix: Union[None, str] = None,
            data_format: str = 'classic',
            batch_size: int = 32,
            drop_last: bool = False,
            input_variables: Union[None, Sequence] = None,
            output_variables: Union[None, Sequence] = None,
            constants: Union[None, DefaultDict] = None,
            scaling: Union[None, DefaultDict] = None,
            splits: Union[None, DefaultDict] = None,
            input_time_dim: int = 1,
            output_time_dim: int = 1,
            data_time_step: Union[int, str] = '3H',
            time_step: Union[int, str] = '6H',
            gap: Union[int, str, None] = None,
            shuffle: bool = True,
            add_insolation: bool = False,
            cube_dim: int = 64,
            num_workers: int = 4,
            pin_memory: bool = True
    ):
        super().__init__()
        self.directory = directory
        self.prefix = prefix
        self.suffix = suffix
        self.data_format = data_format
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.input_variables = input_variables
        self.output_variables = output_variables or input_variables
        self.constants = constants
        self.scaling = scaling
        self.splits = splits
        self.input_time_dim = input_time_dim
        self.output_time_dim = output_time_dim
        self.data_time_step = data_time_step
        self.time_step = time_step
        self.gap = gap
        self.shuffle = shuffle
        self.add_insolation = add_insolation
        self.cube_dim = cube_dim
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.data_format == 'classic':
            dataset = open_time_series_dataset_classic(
                directory=self.directory,
                input_variables=self.input_variables,
                output_variables=self.output_variables,
                constants=self.constants,
                prefix=self.prefix,
                suffix=self.suffix,
                batch_size=self.batch_size
            )
        else:
            raise NotImplementedError
        if self.splits is not None:
            self.train_dataset = TimeSeriesDataset(
                dataset.sel(time=slice(self.splits['train_date_start'], self.splits['train_date_end'])),
                scaling=self.scaling,
                input_time_dim=self.input_time_dim,
                output_time_dim=self.output_time_dim,
                data_time_step=self.data_time_step,
                time_step=self.time_step,
                gap=self.gap,
                batch_size=self.batch_size,
                drop_last=self.drop_last,
                add_insolation=self.add_insolation
            )
            self.val_dataset = TimeSeriesDataset(
                dataset.sel(time=slice(self.splits['val_date_start'], self.splits['val_date_end'])),
                scaling=self.scaling,
                input_time_dim=self.input_time_dim,
                output_time_dim=self.output_time_dim,
                data_time_step=self.data_time_step,
                time_step=self.time_step,
                gap=self.gap,
                batch_size=self.batch_size,
                drop_last=False,
                add_insolation=self.add_insolation
            )
            self.test_dataset = TimeSeriesDataset(
                dataset.sel(time=slice(self.splits['test_date_start'], self.splits['test_date_end'])),
                scaling=self.scaling,
                input_time_dim=self.input_time_dim,
                output_time_dim=self.output_time_dim,
                data_time_step=self.data_time_step,
                time_step=self.time_step,
                gap=self.gap,
                batch_size=self.batch_size,
                drop_last=False,
                add_insolation=self.add_insolation
            )
        else:
            self.test_dataset = TimeSeriesDataset(
                dataset,
                scaling=self.scaling,
                input_time_dim=self.input_time_dim,
                output_time_dim=self.output_time_dim,
                data_time_step=self.data_time_step,
                time_step=self.time_step,
                gap=self.gap,
                batch_size=self.batch_size,
                drop_last=False,
                add_insolation=self.add_insolation
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            batch_size=None
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            shuffle=False,
            batch_size=None
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            shuffle=False,
            batch_size=None
        )

