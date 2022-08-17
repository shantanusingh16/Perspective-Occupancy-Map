import imp

from .indolayout import Indolayout
from models.monolayout import Monolayout
from models.deeplabv3 import DeepLabv3
from models.indolayout import Indolayout

model_classes = {
    'monolayout': Monolayout,
    'deeplabv3': DeepLabv3,
    'indolayout': Indolayout
}

def make_model(cfg):
    return model_classes[cfg.model_type](cfg.learning_rate)
