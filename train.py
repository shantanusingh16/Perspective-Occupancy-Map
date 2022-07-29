import sys

import cprint
import argparse

args = argparse.ArgumentParser()
args.add_argument("--config_path", help="Path to Config File", default="configs/example_config.yaml")
args = args.parse_args()

from configs import get_cfg_defaults

import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything

from torch.utils.data import DataLoader

from dataset import make_dataset
from model import make_model

if __name__ == "__main__":
    
    # Load config file
    cfg = get_cfg_defaults()

    if args.config_path[-4:] == 'yaml':
        cfg.merge_from_file(args.config_path)
    else:
        print("No valid config specified, using default")

    cfg.freeze()
    cprint.info(cfg)

    # Make Dataset
    train_dataset = make_dataset(cfg, is_val=False)
    val_dataset = make_dataset(cfg, is_val=True)

    # Make dataloader
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, 
                            pin_memory=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            pin_memory=True, num_workers=cfg.num_workers)

    # Make model
    device = torch.device("cpu" if cfg.no_cuda else "cuda")
    model = make_model(cfg).to(device)

    cprint.warn("Model made, running on " + "cpu" if cfg.no_cuda else "gpu")

    # Prepare Logging
    wandb_logger = WandbLogger(name=cfg.name, log_model=True)
    wandb_logger.watch(model, log='all')

    # Prepare checkpointing and saving
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        filename='model_name-{epoch:02d}-{val_acc:.2f}',
        save_top_k=3,
        mode='max',
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Train!
    trainer = pl.Trainer(gpus=1, logger=wandb_logger, log_every_n_steps=1,
            callbacks=[checkpoint_callback, lr_monitor], check_val_every_n_epoch=3, max_epochs=30)
    trainer.fit(model, train_loader, val_loader)