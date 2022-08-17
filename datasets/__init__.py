from .gibson4 import Gibson4DataModule

datamodule_classes = {
    'gibson4': Gibson4DataModule
}

def make_datamodule(cfg):
    dm_class = datamodule_classes[cfg.dataset]
    dm = dm_class(cfg.data_dir, cfg.split_dir, cfg.train_val_split, cfg.batch_size, cfg.num_workers)
    return dm