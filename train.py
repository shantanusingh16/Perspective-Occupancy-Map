import os
import time

from cprint import cprint
import argparse

args = argparse.ArgumentParser()
args.add_argument("--config_path", help="Path to Config File", default="configs/example_config.yaml")
args = args.parse_args()

from configs import get_cfg_defaults

import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything

from datasets import make_datamodule
from models import make_model

from pytorch_lightning.profilers import AdvancedProfiler

if __name__ == "__main__":
    
    # Load config file
    cfg = get_cfg_defaults()

    if args.config_path[-4:] == 'yaml':
        cfg.merge_from_file(args.config_path)
    else:
        print("No valid config specified, using default")

    cfg.freeze()
    cprint.info(cfg)

    # Make datamodule
    dm = make_datamodule(cfg)
    cprint.warn(f"Datamodule {cfg.dataset} made")

    # Make model
    model = make_model(cfg)
    cprint.warn(f"Model {cfg.model_type} made, running on " + ("cpu" if cfg.no_cuda else "gpu"))

    # Unique timstamp for logging:
    ts = time.strftime("%Y%m%d-%H%M%S")

    # Prepare Logging
    log_dir = os.path.join(cfg.log_dir, cfg.experiment_name, ts)
    os.makedirs(log_dir)
    logger = WandbLogger(name=cfg.experiment_name, project=cfg.project_name, save_dir=log_dir, log_model=True)
    logger.watch(model, log='all')


    # Prepare checkpointing and saving
    weight_dir = os.path.join(cfg.weight_dir, cfg.experiment_name, ts)
    os.makedirs(weight_dir)
    checkpoint_callback = ModelCheckpoint(
        dirpath=weight_dir,
        monitor='val_loss',
        filename= cfg.experiment_name + '-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
        every_n_epochs=1
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # aprofiler = AdvancedProfiler(filename='perf_logs')

    # Train!
    trainer = pl.Trainer(accelerator='gpu', devices=[0,1], logger=logger, log_every_n_steps=cfg.log_frequency,
            callbacks=[checkpoint_callback, lr_monitor], check_val_every_n_epoch=1, max_epochs=cfg.num_epochs, strategy='ddp')
    trainer.fit(model, dm)

    # Test!
    trainer.test(model, dm)