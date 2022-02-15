import logging
import os
import time
from typing import DefaultDict, Sequence, Union

from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import xarray as xr

from dlwp.utils import insolation

logger = logging.getLogger(__name__)


def open_time_series_dataset_classic(
        directory: str,
        input_variables: Sequence,
        output_variables: Union[Sequence, None],
        constants: Union[DefaultDict, None] = None,
        prefix: Union[str, None] = None,
        suffix: Union[str, None] = None,
        batch_size: int = 32,
) -> xr.Dataset:
    output_variables = output_variables or input_variables
    all_variables = np.intersect1d(input_variables, output_variables)
    prefix = prefix or ''
    suffix = suffix or ''

    def get_file_name(path, var):
        return os.path.join(path, f"{prefix}{var}{suffix}.nc")

    merge_time = time.time()
    logger.info("merging input datasets")

    datasets = []
    remove_attrs = ['varlev', 'mean', 'std']
    for variable in all_variables:
        file_name = get_file_name(directory, variable)
        logger.debug(f"open nc dataset {file_name}")
        ds = xr.open_dataset(file_name, chunks={'sample': batch_size}).isel(varlev=0)
        for attr in remove_attrs:
            try:
                ds = ds.drop(attr)
            except ValueError:
                pass
        # Rename variable
        ds = ds.rename({'predictors': variable, 'sample': 'time'})
        # Change lat/lon to coordinates
        try:
            ds = ds.set_coords(['lat', 'lon'])
        except (ValueError, KeyError):
            pass
        datasets.append(ds)
    # Merge datasets
    data = xr.merge(datasets)

    # Convert to input/target array by merging along the variables
    input_da = data[list(input_variables)].to_array('channel_in', name='inputs').transpose(
        'time', 'channel_in', 'face', 'height', 'width')
    target_da = data[list(output_variables)].to_array('channel_out', name='targets').transpose(
        'time', 'channel_out', 'face', 'height', 'width')

    result = xr.Dataset()
    result['inputs'] = input_da
    result['targets'] = target_da

    # Get constants
    if constants is not None:
        constants_ds = []
        for name, var in constants.items():
            constants_ds.append(xr.open_dataset(get_file_name(directory, name)).set_coords(['lat', 'lon'])[var])
        constants_ds = xr.merge(constants_ds, compat='override')
        constants_da = constants_ds.to_array('channel_c', name='constants').transpose(
            'channel_c', 'face', 'height', 'width')
        result['constants'] = constants_da

    logger.info(f"merged datasets in {time.time() - merge_time:0.1f} s")

    return result


class TimeSeriesDataset(Dataset):
    def __init__(
            self,
            dataset: xr.Dataset,
            scaling: DefaultDict,
            input_time_dim: int = 1,
            output_time_dim: int = 1,
            data_time_step: Union[int, str] = '3H',
            time_step: Union[int, str] = '6H',
            gap: Union[int, str, None] = None,
            batch_size: int = 32,
            drop_last: bool = False,
            add_insolation: bool = False
    ):
        self.ds = dataset
        self.scaling = OmegaConf.to_object(scaling)
        self.input_time_dim = input_time_dim
        self.output_time_dim = output_time_dim
        self.data_time_step = self._convert_time_step(data_time_step)
        self.time_step = self._convert_time_step(time_step)
        self.gap = self._convert_time_step(gap if gap is not None else time_step)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.add_insolation = add_insolation

        # Time stepping
        if (self.time_step % self.data_time_step).total_seconds() != 0:
            raise ValueError(f"'time_step' must be a multiple of 'data_time_step' "
                             f"(got {self.time_step} and {self.data_time_step}")
        if (self.gap % self.data_time_step).total_seconds() != 0:
            raise ValueError(f"'gap' must be a multiple of 'data_time_step' "
                             f"(got {self.gap} and {self.data_time_step}")
        self._interval = self.time_step // self.data_time_step
        self._window_length = (
            self._interval * (self.input_time_dim - 1) + 1 +
            (self.gap // self.data_time_step) +
            self._interval * (self.output_time_dim - 1)  # first point is counted by gap
        )
        self._batch_window_length = self.batch_size + self._window_length - 1
        self._output_delay = self._interval * (self.input_time_dim - 1) + (self.gap // self.data_time_step)
        # Indices within a batch
        self._input_indices = [list(range(n, n + self._interval * self.input_time_dim, self._interval))
                               for n in range(self.batch_size)]
        self._output_indices = [list(range(n + self._output_delay,
                                           n + self._interval * self.output_time_dim + self._output_delay,
                                           self._interval))
                               for n in range(self.batch_size)]

        self._spatial_dims = (self.ds.dims['face'], self.ds.dims['height'], self.ds.dims['width'])

        self._input_scaling = None
        self._target_scaling = None
        self._get_scaling_da()

    @staticmethod
    def _convert_time_step(dt):
        return pd.Timedelta(hours=dt) if isinstance(dt, int) else pd.Timedelta(dt)

    def _get_scaling_da(self):
        scaling_df = pd.DataFrame.from_dict(self.scaling).T
        scaling_df.loc['zeros'] = {'mean': 0., 'std': 1.}
        scaling_da = scaling_df.to_xarray().astype('float32')
        try:
            self._input_scaling = scaling_da.sel(index=self.ds.channel_in.values).rename({'index': 'channel_in'})
        except (ValueError, KeyError):
            raise KeyError(f"one or more of the input data variables f{list(self.ds.channel_in)} not found in the "
                           f"scaling config dict data.scaling ({list(self.scaling.keys())})")
        try:
            self._target_scaling = scaling_da.sel(index=self.ds.channel_out.values).rename({'index': 'channel_out'})
        except (ValueError, KeyError):
            raise KeyError(f"one or more of the target data variables f{list(self.ds.channel_out)} not found in the "
                           f"scaling config dict data.scaling ({list(self.scaling.keys())})")

    def __len__(self):
        length = (self.ds.dims['time'] - self._window_length + 1) / self.batch_size
        if self.drop_last:
            return int(np.floor(length))
        else:
            return int(np.ceil(length))

    def __get_time_slice(self, item):
        max_index = (item + 1) * self.batch_size + self._window_length
        if not self.drop_last and max_index > self.ds.dims['time']:
            batch_size = self.batch_size - (max_index - self.ds.dims['time'])
        else:
            batch_size = self.batch_size
        return slice(item * self.batch_size, max_index), batch_size

    def __getitem__(self, item):
        if item < 0:
            item = len(self) + item
        time_slice, this_batch = self.__get_time_slice(item)
        batch = {'time': time_slice}
        load_time = time.time()
        input_array = ((self.ds['inputs'].isel(**batch) - self._input_scaling['mean']) /
                       self._input_scaling['std']).compute()
        target_array = ((self.ds['targets'].isel(**batch) - self._target_scaling['mean']) /
                        self._target_scaling['std']).compute()
        logger.log(5, f"loaded batch data in {time.time() - load_time:0.2f} s")

        compute_time = time.time()
        # Insolation
        if self.add_insolation:
            sol = insolation(self.ds.time[time_slice].values, self.ds.lat.values, self.ds.lon.values)[:, None]
            decoder_inputs = np.empty((this_batch, self.input_time_dim + self.output_time_dim, 1) +
                                      self._spatial_dims, dtype='float32')

        # Get buffers for the batches, which we'll fill in iteratively.
        inputs = np.empty((this_batch, self.input_time_dim, self.ds.dims['channel_in']) +
                          self._spatial_dims, dtype='float32')
        targets = np.empty((this_batch, self.output_time_dim, self.ds.dims['channel_out']) +
                           self._spatial_dims, dtype='float32')

        for sample in range(this_batch):
            inputs[sample] = input_array[self._input_indices[sample]]
            targets[sample] = target_array[self._output_indices[sample]]
            if self.add_insolation:
                decoder_inputs[sample] = sol[self._input_indices[sample] + self._output_indices[sample]]

        inputs_result = [inputs]
        if self.add_insolation:
            inputs_result.append(decoder_inputs)
        if 'constants' in self.ds.data_vars:
            inputs_result.append(self.ds.constants.values)
        logger.log(5, f"computed batch in {time.time() - compute_time:0.2f} s")

        return inputs_result, targets
