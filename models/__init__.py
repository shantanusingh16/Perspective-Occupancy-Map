from pytorch_lightning import LightningModule
from models.amodal_bev import AmodalBev
from models.monolayout import Monolayout
from models.deeplabv3 import DeepLabv3
from models.indolayout import Indolayout
from models.pom import POMv1, POMv2
from models.pon import PON
from models.tim import TIM

model_classes = {
    'monolayout': Monolayout,
    'deeplabv3': DeepLabv3,
    'indolayout': Indolayout,
    'amodalbev': AmodalBev,
    'pon': PON,
    'tim': TIM,
    'pomv1': POMv1,
    'pomv2': POMv2
}

def make_model(cfg) -> LightningModule:
    return model_classes[cfg.model_type](cfg.learning_rate, **cfg.model_hparams)
