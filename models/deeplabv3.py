import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from utils import logging 
from torchvision.transforms import Resize, Normalize, Compose, InterpolationMode
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
import torchmetrics

'''
URLs:
https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_resnet101.html#torchvision.models.segmentation.DeepLabV3_ResNet101_Weights
'''


class DeepLabv3(pl.LightningModule):
    def __init__(self, learning_rate):
        super().__init__()
        self.model = deeplabv3_mobilenet_v3_large(weights=None, num_classes=3)
        self.model_transforms = Compose([Resize(520), Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        #TODO: Fix hardcoded outputsize
        self.output_transform = Resize((128, 128), interpolation=InterpolationMode.NEAREST)
        self.learning_rate = learning_rate

        self.save_hyperparameters()

    def setup(self, stage= None) -> None:
        if stage == 'fit' or stage is None:
            self.val_metrics = torchmetrics.MetricCollection([torchmetrics.JaccardIndex(3, average=None), \
                torchmetrics.AveragePrecision(3, average=None)])

            ## TODO: Change hard-coded image log frequency to read from config
            self.val_img_logger = logging.get_segout_logger(self.logger.experiment)


    def forward(self, x):
        x = self.model_transforms(x)
        y_hat = self.model(x)['out']
        y_hat = self.output_transform(y_hat)
        return y_hat

    def training_step(self, batch, batch_idx):
        rgb, _, pom, _ = batch
        y_hat = self(rgb)
        loss = F.cross_entropy(y_hat, pom.squeeze().long())
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self._shared_eval(batch, batch_idx, "val")
        
        y_gt = batch[2].squeeze().long()
        rgb = batch[0]

        self.val_metrics.update(y_hat, y_gt)
        self.val_img_logger.log_image(X=rgb, pred=y_hat, gt=y_gt, batch_idx=batch_idx)
       

    def on_validation_epoch_end(self) -> None:
        self.val_img_logger.flush()

        iu, ap =  self.val_metrics.compute().values()
        self.log('mIOU', {'unknown': iu[0], 'occupied': iu[1], 'free': iu[2]}, sync_dist=True)
        self.log('mAP', {'unknown': ap[0], 'occupied': ap[1], 'free': ap[2]}, sync_dist=True)

        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        rgb, _, _, _ = batch
        y_hat = self(rgb)
        return torch.argmax(y_hat, dim=1)

    def _shared_eval(self, batch, batch_idx, prefix):
        rgb, _, pom, _ = batch
        y_hat = self(rgb)
        loss = F.cross_entropy(y_hat, pom.squeeze().long())
        self.log(f"{prefix}_loss", loss, sync_dist=True)
        return y_hat


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)



if __name__ == "__main__":
    model = DeepLabv3()

    input_rgb = torch.rand((4, 3, 512, 512))
    lbl = torch.randint(3, size=((4, 1, 128, 128))).long()
    model.training_step((input_rgb, None, lbl, None), 0)