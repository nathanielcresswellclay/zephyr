import logging
import argparse
import os
from pathlib import Path
import time

from hydra import initialize, compose
from hydra.utils import instantiate
import dask.array
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import xarray as xr

from dlwp.utils import to_chunked_dataset, encode_variables_as_int, configure_logging

logger = logging.getLogger(__name__)
logging.getLogger('cfgrib').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)


def _convert_time_step(dt):
    return pd.Timedelta(hours=dt) if isinstance(dt, int) else pd.Timedelta(dt)


def get_forecast_dates(start, end, freq):
    # Get the dates parameters
    if freq == 'biweekly':
        dates_1 = pd.date_range(start, end, freq='7D')
        dates_2 = pd.date_range(pd.Timestamp(start) + pd.Timedelta(days=3), end, freq='7D')
        dates = dates_1.append(dates_2).sort_values().to_numpy()
    else:
        dates = pd.date_range(start, end, freq=freq).to_numpy()
    return dates


def inference(args: argparse.Namespace):
    forecast_dates = get_forecast_dates(args.forecast_init_start, args.forecast_init_end, args.freq)
    os.makedirs(args.output_directory, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with initialize(config_path=os.path.join(args.hydra_path, '.hydra')):
        cfg = compose('config.yaml')

    # Set up data module with some overrides for inference. Compute expected output time dimension.
    output_lead_times = np.arange(
        _convert_time_step(cfg.data.gap),
        _convert_time_step(args.lead_time) + pd.Timedelta(seconds=1),
        _convert_time_step(cfg.data.time_step)
    )
    output_time_dim = len(output_lead_times)
    optional_kwargs = {k: v for k, v in {
        'directory': args.data_directory,
        'prefix': args.data_prefix,
        'suffix': args.data_suffix
    }.items() if v is not None}
    data_module = instantiate(
        cfg.data.module,
        output_time_dim=output_time_dim,
        forecast_init_times=forecast_dates,
        shuffle=False,
        batch_size=1,
        **optional_kwargs
    )
    data_module.setup()
    loader = data_module.test_dataloader()

    # Load the model checkpoint. Set output_time_dim param override.
    model_name = Path(args.model_path).name
    model = instantiate(cfg.model)
    checkpoint = os.path.join(args.model_path, 'tensorboard', model_name, f'version_{args.model_version}',
                              'checkpoints', args.model_checkpoint)
    logger.info(f"load model checkpoint {checkpoint}")
    model = model.load_from_checkpoint(checkpoint, map_location=device, output_time_dim=output_time_dim)
    model = model.to(device)

    # Allocate giant array. One extra time step for the init state.
    logger.info("allocating prediction array. If this fails due to OOM consider reducing lead times or "
                "number of forecasts.")
    prediction = np.empty(
        (len(loader), output_time_dim + 1, len(data_module.output_variables)) + data_module.test_dataset.spatial_dims,
        dtype='float32'
    )

    # Iterate over model predictions
    pbar = tqdm(loader)
    for b, inputs in enumerate(pbar):
        pbar.postfix = pd.Timestamp(forecast_dates[b]).strftime('init %Y-%m-%d %HZ')
        pbar.update()
        # Last input time step for init state
        prediction[b][0] = inputs[0][0, -1]
        inputs = [i.to(device) for i in inputs]
        with torch.no_grad():
            prediction[b][1:] = model(inputs).cpu().numpy()

    # Generate dataarray with coordinates
    meta_ds = data_module.test_dataset.ds
    if args.to_zarr:
        prediction = dask.array.from_array(prediction, chunks=(1,) + prediction.shape[1:])
    prediction_da = xr.DataArray(
        prediction,
        dims=['time', 'step', 'channel_out', 'face', 'height', 'width'],
        coords={
            'time': forecast_dates,
            'step': [pd.Timedelta(hours=0)] + list(output_lead_times),
            'channel_out': cfg.data.output_variables or cfg.data.input_variables,
            'face': meta_ds.face,
            'height': meta_ds.height,
            'width': meta_ds.width
        }
    )
    # Re-scale prediction
    prediction_da[:] *= data_module.test_dataset.target_scaling['std']
    prediction_da[:] += data_module.test_dataset.target_scaling['mean']
    prediction_ds = prediction_da.to_dataset(dim='channel_out')
    for variable in prediction_ds.data_vars:
        if cfg.data.scaling[variable].get('log_epsilon', None) is not None:
            prediction_ds[variable] = np.exp(
                prediction_ds[variable] + np.log(cfg.data.scaling[variable]['log_epsilon'])
            ) - cfg.data.scaling[variable]['log_epsilon']

    # Export dataset
    write_time = time.time()
    prediction_ds = to_chunked_dataset(prediction_ds, {'time': 1})
    if args.encode_int:
        prediction_ds = encode_variables_as_int(prediction_ds, compress=1)

    output_file = os.path.join(args.output_directory,
                               f"forecast_{model_name}_v{args.model_version}.{'zarr' if args.to_zarr else 'nc'}")
    logger.info(f"exporting data to {output_file}")
    if args.to_zarr:
        prediction_ds.to_zarr(output_file)
    else:
        prediction_ds.to_netcdf(output_file)
    logger.debug(f"wrote file in {time.time() - write_time:0.1f} s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Produce forecasts from a DLWP model.')
    parser.add_argument('-m', '--model-path', type=str, required=True,
                        help="Path to model training outputs directory")
    parser.add_argument('-c', '--model-checkpoint', type=str, default='last.ckpt',
                        help="Model checkpoint file name")
    parser.add_argument('--model-version', default=0, type=int,
                        help="Model version. Should usually be 0 unless an explicit re-start in the same output "
                             "directory was made.")
    parser.add_argument('-l', '--lead-time', type=int, default=168,
                        help="Maximum forecast lead time to predict, in integer hours")
    parser.add_argument('-s', '--forecast-init-start', type=str, default='2017-01-02',
                        help="")
    parser.add_argument('-e', '--forecast-init-end', type=str, default='2018-12-30',
                        help="")
    parser.add_argument('-f', '--freq', type=str, default='biweekly',
                        help="Frequency of forecast initialization. There is a special case, 'biweekly', which will "
                             "follow the ECMWF standard of two forecasts per week, with a 3- followed by 4-day gap. "
                             "Otherwise, interpretable by pandas.")
    parser.add_argument('-o', '--output-directory', type=str, default='.',
                        help="Directory in which to save output forecast")
    parser.add_argument('--encode-int', action='store_true',
                        help="Encode data variables as int16 type (may not be compatible with tempest-remap)")
    parser.add_argument('--to-zarr', action='store_true',
                        help="Export data in zarr format")
    parser.add_argument('-d', '--data-directory', type=str, default=None,
                        help="Path to test data, if different from files used for model training")
    parser.add_argument('--data-prefix', type=str, default=None,
                        help="Prefix for test data files. "
                             "Assumes files of naming scheme {prefix}{var}_{lead}h{suffix}.zarr")
    parser.add_argument('--data-suffix', type=str, default=None,
                        help="Suffix for test data files. "
                             "Assumes files of naming scheme {prefix}{var}_{lead}h{suffix}.zarr")

    configure_logging()
    run_args = parser.parse_args()

    # Hydra requires a relative (not absolute) path to working config directory. It also works in a sub-directory of
    # current python working directory.
    run_args.hydra_path = os.path.relpath(run_args.model_path, os.path.join(os.getcwd(), 'hydra'))
    logger.debug(f"model path: {run_args.model_path}")
    logger.debug(f"python working dir: {os.getcwd()}")
    logger.debug(f"hydra path: {run_args.hydra_path}")
    inference(run_args)
