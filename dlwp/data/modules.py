# System modules
import logging
from abc import ABC
from typing import Optional, Union, Sequence

# External modules
from omegaconf import DictConfig
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# Internal modules
from .data_loading import open_time_series_dataset_classic, TimeSeriesDataset

logger = logging.getLogger(__name__)


class TimeSeriesDataModule(pl.LightningDataModule, ABC):
    def __init__(
            self,
            directory: str = '.',
            prefix: Optional[str] = None,
            suffix: Optional[str] = None,
            data_format: str = 'classic',
            batch_size: int = 32,
            drop_last: bool = False,
            input_variables: Optional[Sequence] = None,
            output_variables: Optional[Sequence] = None,
            constants: Optional[DictConfig] = None,
            scaling: Optional[DictConfig] = None,
            splits: Optional[DictConfig] = None,
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
        """
        pytorch-lightning module for complete model train, validation, and test data loading. Uses
        dlwp.data.data_loading.TimeSeriesDataset under-the-hood. Loaded data files follow the naming scheme
            {directory}/{prefix}{variable/constant}{suffix}{[.nc, .zarr]}

        :param directory: directory containing data files
        :param prefix: prefix appended to all data files
        :param suffix: suffix appended to all data files
        :param data_format: currently only 'classic' is allowed.
            'classic': use classic DLWP file types. Loads .nc files, assuming dimensions [sample, varlev, face, height,
                width] and data variables 'predictors', 'lat', and 'lon'.
        :param batch_size: size of batches to draw from data
        :param drop_last: whether to drop the last batch if it is smaller than batch_size
        :param input_variables: list of input variable names, to be found in data file name
        :param output_variables: list of output variables names. If None, defaults to `input_variables`.
        :param constants: dictionary with {key: value} corresponding to {constant_name: variable name in file}.
        :param scaling: dictionary containing scaling parameters for data variables
        :param splits: dictionary with train/validation/test set start/end dates. If not provided, loads the entire
            data time series as the test set.
        :param input_time_dim: number of time steps in the input array
        :param output_time_dim: number of time steps in the output array
        :param data_time_step: either integer hours or a str interpretable by pandas: time between steps in the
            original data time series
        :param time_step: either integer hours or a str interpretable by pandas: desired time between effective model
            time steps
        :param gap: either integer hours or a str interpretable by pandas: time step between the last input time and
            the first output time. Defaults to `time_step`.
        :param shuffle: option to shuffle the training data
        :param add_insolation: option to add prescribed insolation as a decoder input feature
        :param cube_dim: number of points on the side of a cube face. Not currently used.
        :param num_workers: number of parallel data loading workers
        :param pin_memory: enable pytorch's memory pinning for faster GPU I/O
        """
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

