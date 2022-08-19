import os
import time

from cprint import cprint
import argparse

args = argparse.ArgumentParser()
args.add_argument("--config_path", help="Path to Config File", default="configs/example_config.yaml")
args = args.parse_args()

from configs import get_cfg_defaults

import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger

from datasets import make_datamodule
from models import make_model

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

    # Load model weights from checkpoint point
    assert os.path.exists(cfg.load_ckpt_path)
    model = model.load_from_checkpoint(cfg.load_ckpt_path)

    # Unique timstamp for logging:
    ts = time.strftime("%Y%m%d-%H%M%S")

    # Prepare Logging
    log_dir = os.path.join(cfg.log_dir, cfg.experiment_name, ts)
    os.makedirs(log_dir)
    logger = WandbLogger(name=cfg.experiment_name, project=cfg.project_name, save_dir=log_dir, log_model=False)

    # Test!
    trainer = pl.Trainer(accelerator='gpu', devices=[0], logger=logger, enable_checkpointing=False)
    trainer.test(model, dm)