import logging
import os
import time
from typing import DefaultDict, Optional, Sequence, Union

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
        output_variables: Optional[Sequence],
        constants: Optional[DefaultDict] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        batch_size: int = 32,
        scaling: Optional[DictConfig] = None
) -> xr.Dataset:
    output_variables = output_variables or input_variables
    all_variables = np.union1d(input_variables, output_variables)
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
        logger.debug("open nc dataset %s", file_name)
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
        # Apply log scaling lazily
        if variable in scaling and scaling[variable].get('log_epsilon', None) is not None:
            ds[variable] = np.log(ds[variable] + scaling[variable]['log_epsilon']) \
                           - np.log(scaling[variable]['log_epsilon'])
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

    logger.info("merged datasets in %0.1f s", time.time() - merge_time)

    return result


def open_time_series_dataset_zarr(
        directory: str,
        input_variables: Sequence,
        output_variables: Optional[Sequence],
        constants: Optional[DefaultDict] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        batch_size: int = 32,  # pylint: disable=unused-argument
        scaling: Optional[DictConfig] = None
) -> xr.Dataset:
    output_variables = output_variables or input_variables
    all_variables = np.union1d(input_variables, output_variables)
    prefix = prefix or ''
    suffix = suffix or ''

    def get_file_name(path, var):
        return os.path.join(path, f"{prefix}{var}{suffix}.zarr")

    merge_time = time.time()
    logger.info("merging input datasets")

    datasets = []
    for variable in all_variables:
        file_name = get_file_name(directory, variable)
        logger.debug("open zarr dataset %s", file_name)
        ds = xr.open_zarr(file_name)
        # Apply log scaling lazily
        if variable in scaling and scaling[variable].get('log_epsilon', None) is not None:
            ds[variable] = np.log(ds[variable] + scaling[variable]['log_epsilon']) \
                           - np.log(scaling[variable]['log_epsilon'])
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
            constants_ds.append(xr.open_zarr(get_file_name(directory, name))[var])
        constants_ds = xr.merge(constants_ds, compat='override')
        constants_da = constants_ds.to_array('channel_c', name='constants').transpose(
            'channel_c', 'face', 'height', 'width')
        result['constants'] = constants_da

    logger.info("merged datasets in %0.1f s", time.time() - merge_time)

    return result


class TimeSeriesDataset(Dataset):
    def __init__(
            self,
            dataset: xr.Dataset,
            scaling: DictConfig,
            input_time_dim: int = 1,
            output_time_dim: int = 1,
            data_time_step: Union[int, str] = '3H',
            time_step: Union[int, str] = '6H',
            gap: Union[int, str, None] = None,
            batch_size: int = 32,
            drop_last: bool = False,
            add_insolation: bool = False,
            forecast_init_times: Optional[Sequence] = None
    ):
        """
        Dataset for sampling from continuous time-series data, compatible with pytorch data loading.

        :param dataset: xarray Dataset produced by one of the `open_*` methods herein
        :param scaling: dictionary containing scaling parameters for data variables
        :param input_time_dim: number of time steps in the input array
        :param output_time_dim: number of time steps in the output array
        :param data_time_step: either integer hours or a str interpretable by pandas: time between steps in the
            original data time series
        :param time_step: either integer hours or a str interpretable by pandas: desired time between effective model
            time steps
        :param gap: either integer hours or a str interpretable by pandas: time step between the last input time and
            the first output time. Defaults to `time_step`.
        :param batch_size: batch size
        :param drop_last: whether to drop the last batch if it is smaller than batch_size
        :param add_insolation: option to add prescribed insolation as a decoder input feature
        :param forecast_init_times: a Sequence of pandas Timestamps dictating the specific initialization times
            to produce inputs for. Note that providing this parameter configures the data loader to only produce
            this number of samples, and NOT produce any target array.
        """
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
        self.forecast_init_times = forecast_init_times
        self.forecast_mode = self.forecast_init_times is not None

        # Time stepping
        if (self.time_step % self.data_time_step).total_seconds() != 0:
            raise ValueError(f"'time_step' must be a multiple of 'data_time_step' "
                             f"(got {self.time_step} and {self.data_time_step}")
        if (self.gap % self.data_time_step).total_seconds() != 0:
            raise ValueError(f"'gap' must be a multiple of 'data_time_step' "
                             f"(got {self.gap} and {self.data_time_step}")
        self.interval = self.time_step // self.data_time_step

        # Find indices of init times for forecast mode
        if self.forecast_mode:
            if self.batch_size != 1:
                self.batch_size = 1
                logger.warning("providing 'forecast_init_times' to TimeSeriesDataset requires `batch_size=1`; "
                               "setting it now")
            self._forecast_init_indices = np.array(
                [int(np.where(self.ds['time'] == s)[0]) for s in self.forecast_init_times],
                dtype='int'
            ) - ((self.input_time_dim - 1) * self.interval)
        else:
            self._forecast_init_indices = None

        # Length of the data window needed for one sample.
        if self.forecast_mode:
            self._window_length = self.interval * (self.input_time_dim - 1) + 1
        else:
            self._window_length = (
                    self.interval * (self.input_time_dim - 1) + 1 +
                    (self.gap // self.data_time_step) +
                    self.interval * (self.output_time_dim - 1)  # first point is counted by gap
            )
        self._batch_window_length = self.batch_size + self._window_length - 1
        self._output_delay = self.interval * (self.input_time_dim - 1) + (self.gap // self.data_time_step)
        # Indices within a batch
        self._input_indices = [list(range(n, n + self.interval * self.input_time_dim, self.interval))
                               for n in range(self.batch_size)]
        self._output_indices = [list(range(n + self._output_delay,
                                           n + self.interval * self.output_time_dim + self._output_delay,
                                           self.interval))
                               for n in range(self.batch_size)]

        self.spatial_dims = (self.ds.dims['face'], self.ds.dims['height'], self.ds.dims['width'])

        self.input_scaling = None
        self.target_scaling = None
        self._get_scaling_da()

    @staticmethod
    def _convert_time_step(dt):  # pylint: disable=invalid-name
        return pd.Timedelta(hours=dt) if isinstance(dt, int) else pd.Timedelta(dt)

    def _get_scaling_da(self):
        scaling_df = pd.DataFrame.from_dict(self.scaling).T
        scaling_df.loc['zeros'] = {'mean': 0., 'std': 1.}
        scaling_da = scaling_df.to_xarray().astype('float32')
        try:
            self.input_scaling = scaling_da.sel(index=self.ds.channel_in.values).rename({'index': 'channel_in'})
        except (ValueError, KeyError):
            raise KeyError(f"one or more of the input data variables f{list(self.ds.channel_in)} not found in the "
                           f"scaling config dict data.scaling ({list(self.scaling.keys())})")
        try:
            self.target_scaling = scaling_da.sel(index=self.ds.channel_out.values).rename({'index': 'channel_out'})
        except (ValueError, KeyError):
            raise KeyError(f"one or more of the target data variables f{list(self.ds.channel_out)} not found in the "
                           f"scaling config dict data.scaling ({list(self.scaling.keys())})")

    def __len__(self):
        if self.forecast_mode:
            return len(self._forecast_init_indices)
        length = (self.ds.dims['time'] - self._window_length + 1) / self.batch_size
        if self.drop_last:
            return int(np.floor(length))
        return int(np.ceil(length))

    def _get_time_index(self, item):
        start_index = self._forecast_init_indices[item] if self.forecast_mode else item * self.batch_size
        # TODO: I think this should be -1 and still work (currently missing the last sample in last batch)
        max_index = start_index + self._window_length if self.forecast_mode else \
            (item + 1) * self.batch_size + self._window_length
        if not self.drop_last and max_index > self.ds.dims['time']:
            batch_size = self.batch_size - (max_index - self.ds.dims['time'])
        else:
            batch_size = self.batch_size
        return (start_index, max_index), batch_size

    def _get_forecast_sol_times(self, item):
        time_index, _ = self._get_time_index(item)
        if self.forecast_mode:
            timedeltas = np.array(self._input_indices[0] + self._output_indices[0]) * self.data_time_step
            return self.ds.time[time_index[0]].values + timedeltas
        return self.ds.time[slice(*time_index)].values

    def __getitem__(self, item):
        if item < 0:
            item = len(self) + item
        if item < 0 or item > len(self):
            raise IndexError(f"index {item} out of range for dataset with length {len(self)}")

        time_index, this_batch = self._get_time_index(item)
        batch = {'time': slice(*time_index)}
        load_time = time.time()
        input_array = ((self.ds['inputs'].isel(**batch) - self.input_scaling['mean']) /
                       self.input_scaling['std']).compute()
        if not self.forecast_mode:
            target_array = ((self.ds['targets'].isel(**batch) - self.target_scaling['mean']) /
                            self.target_scaling['std']).compute()
        logger.log(5, "loaded batch data in %0.2f s", time.time() - load_time)

        compute_time = time.time()
        # Insolation
        if self.add_insolation:
            sol = insolation(self._get_forecast_sol_times(item), self.ds.lat.values, self.ds.lon.values)[:, None]
            decoder_inputs = np.empty((this_batch, self.input_time_dim + self.output_time_dim, 1) +
                                      self.spatial_dims, dtype='float32')

        # Get buffers for the batches, which we'll fill in iteratively.
        inputs = np.empty((this_batch, self.input_time_dim, self.ds.dims['channel_in']) +
                          self.spatial_dims, dtype='float32')
        if not self.forecast_mode:
            targets = np.empty((this_batch, self.output_time_dim, self.ds.dims['channel_out']) +
                               self.spatial_dims, dtype='float32')

        # Iterate over valid sample windows
        for sample in range(this_batch):
            inputs[sample] = input_array[self._input_indices[sample]]
            if not self.forecast_mode:
                targets[sample] = target_array[self._output_indices[sample]]
            if self.add_insolation:
                decoder_inputs[sample] = sol if self.forecast_mode else \
                    sol[self._input_indices[sample] + self._output_indices[sample]]

        inputs_result = [inputs]
        if self.add_insolation:
            inputs_result.append(decoder_inputs)
        if 'constants' in self.ds.data_vars:
            inputs_result.append(self.ds.constants.values)
        logger.log(5, "computed batch in %0.2f s", time.time() - compute_time)

        if self.forecast_mode:
            return inputs_result
        return inputs_result, targets
