import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from utils import logging 
from models.encoder_decoder import UNetEncoder, UNetDecoder
import torchmetrics

'''
URLs:
https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_resnet101.html#torchvision.models.segmentation.DeepLabV3_ResNet101_Weights
'''


class AmodalBev(pl.LightningModule):
    def __init__(self, learning_rate, *args, **kwargs):
        super().__init__()
        self.encoder = UNetEncoder(3, 16)
        self.decoder = UNetDecoder(3, 16)
        self.learning_rate = learning_rate

        self.save_hyperparameters()

    def setup(self, stage= None) -> None:
        if (stage == 'fit') or (stage == 'test') or (stage is None):
            self.eval_metrics = torchmetrics.MetricCollection([torchmetrics.JaccardIndex(3, average=None), \
                torchmetrics.AveragePrecision(3, average=None)])

            ## TODO: Change hard-coded image log frequency to read from config
            self.eval_img_logger = logging.get_genout_logger(self.logger.experiment)


    def forward(self, x):
        z_hat = self.encoder(x)
        y_hat = self.decoder(z_hat)
        return y_hat

    def training_step(self, batch, batch_idx):
        _, _, _, bev, proj_bev = batch
        y_hat = self(proj_bev)
        y_gt = bev[:,:,64:,32:96]
        loss = F.cross_entropy(y_hat, y_gt)
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        rgb, _, _, bev, projbev = batch
        y_hat = self(projbev)
        y_gt = bev[:,:,64:,32:96]
        loss = F.cross_entropy(y_hat, y_gt)
        self.log(f"{prefix}_loss", loss, sync_dist=True)

        y_gt = torch.argmax(y_gt, dim=1)
        self.eval_metrics.update(y_hat, y_gt)

        y_hat = torch.argmax(y_hat, dim=1).to(torch.uint8) * 127
        y_gt = y_gt.to(torch.uint8) * 127

        self.eval_img_logger.log_image(X=rgb, pred=y_hat, gt=y_gt, batch_idx=batch_idx)
       
    def on_validation_epoch_end(self) -> None:
        self.log_metrics_and_outputs(stage='val')

    def on_test_epoch_end(self) -> None:
        self.log_metrics_and_outputs(stage='test')

    def log_metrics_and_outputs(self, stage):
        self.eval_img_logger.flush(stage)

        iu, ap =  self.eval_metrics.compute().values()
        self.log(f'{stage}/mIOU', {'unknown': iu[0], 'occupied': iu[1], 'free': iu[2]}, sync_dist=True)
        self.log(f'{stage}/mAP', {'unknown': ap[0], 'occupied': ap[1], 'free': ap[2]}, sync_dist=True)

        self.eval_metrics.reset()

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        _, _, _, _, projbev = batch
        y_hat = self(projbev)
        return torch.argmax(y_hat, dim=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)



if __name__ == "__main__":
    model = AmodalBev(1e-4)

    input_rgb = torch.rand((4, 3, 512, 512))
    bev = torch.randint(1, size=((4, 3, 128, 128))).float()
    projbev = torch.randint(1, size=((4, 3, 64, 64))).float()

    print(bev.unique(), projbev.unique())

    loss = model.training_step((input_rgb, None, None, bev, projbev), 0)
    print(loss)