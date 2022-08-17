import wandb
import pytorch_lightning
from utils.logging import lwandb

'''
For DDP, only rank0 processes are able to log images for now. The other processes use the dummy logger. 
Depending on how the sharding happens, the log after every epoch and for every model might differ in the indices being logged.
To compare models, recommend using single gpu instead of DDP for evaluation runs.
'''


class DummyLogger():
    def __getattribute__(self, __name: str):
        return lambda *args, **kwargs: None


def get_segout_logger(logger):
    '''
    Function to get logging instance to log segmentation outputs.
    The output can serve as a mask for the input and can be overlayed in certain logging tools like WandB.
    '''
    if isinstance(logger, wandb.sdk.wandb_run.Run):
        return lwandb.SegOutLogger(logger)
    if isinstance(logger, pytorch_lightning.loggers.logger.DummyExperiment):
        return DummyLogger()

    raise NotImplementedError(type(logger).__str__)


def get_genout_logger(logger):
    '''
    Function to get logging instance to generated image outputs.
    '''
    if isinstance(logger, wandb.sdk.wandb_run.Run):
        return lwandb.GenOutLogger(logger)
    if isinstance(logger, pytorch_lightning.loggers.logger.DummyExperiment):
        return DummyLogger()

    raise NotImplementedError(type(logger).__str__)