# Default Configuration values and structure of Configuration Node

from yacs.config import CfgNode as CN

_C = CN()

# Project level configuration
_C.project_name = 'perspective_occupancy_map'


# Log configuration
_C.experiment_name = None
_C.log_dir = '/tmp/perspective_occupancy_map/logs'
_C.weight_dir = '/tmp/perspective_occupancy_map/weights'
_C.log_frequency = 250
_C.model_type = None
_C.save_frequency = 1
_C.script_mode = 'train'  # train, eval, predict

# Dataset configuration
_C.dataset = 'gibson4'
_C.data_dir = ''
_C.split_dir = ''
_C.width = 128
_C.height = 128

# Model configuration
_C.load_weights_folder = None

# Training hyperparameters
_C.no_cuda = False
_C.batch_size = 4
_C.train_val_split = 0.9
_C.num_epochs = 100
_C.num_workers = 4
_C.learning_rate = 1e-4
_C.scheduler_step_size = 15
_C.seed = 0


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()
