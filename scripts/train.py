import logging
import os

import hydra
from hydra.utils import instantiate
import pytorch_lightning as pl

logger = logging.getLogger(__name__)
logging.getLogger('cfgrib').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)


@hydra.main(config_path='../configs', config_name='config')
def train(cfg):
    logger.info(f"experiment working directory: {os.getcwd()}")

    # Data module
    data_module = instantiate(cfg.data.module)

    # Model
    input_channels = len(cfg.data.input_variables)
    output_channels = len(cfg.data.output_variables) if cfg.data.output_variables is not None else input_channels
    n_constants = len(cfg.data.get('constants', {}))
    decoder_input_channels = int(cfg.data.get('add_insolation', 0))
    model = instantiate(
        cfg.model,
        input_channels=input_channels,
        output_channels=output_channels,
        n_constants=n_constants,
        decoder_input_channels=decoder_input_channels
    )
    model.hparams['batch_size'] = cfg.batch_size
    print(pl.utilities.model_summary.summarize(model, max_depth=-1))

    # Callbacks for trainer
    if cfg.callbacks is not None:
        callbacks = []
        for _, callback_cfg in cfg.callbacks.items():
            callbacks.append(instantiate(callback_cfg))
    else:
        callbacks = None

    # Logging for trainer
    training_logger = None
    if cfg.logger is not None:
        training_logger = instantiate(cfg.logger, name=os.path.basename(os.getcwd()))

    # Trainer fit
    trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=training_logger)
    trainer.fit(model=model, datamodule=data_module)


if __name__ == '__main__':
    train()
