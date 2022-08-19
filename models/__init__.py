from pytorch_lightning import LightningModule
from models.amodal_bev import AmodalBev
from models.monolayout import Monolayout
from models.deeplabv3 import DeepLabv3
from models.indolayout import Indolayout

model_classes = {
    'monolayout': Monolayout,
    'deeplabv3': DeepLabv3,
    'indolayout': Indolayout,
    'amodalbev': AmodalBev
}

def make_model(cfg) -> LightningModule:
    return model_classes[cfg.model_type](cfg.learning_rate, **cfg.model_hparams)
