import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
from models.encoder_decoder import Encoder, Decoder
from utils.layout_utils import mean_IU, mean_precision
import wandb
from torchvision.transforms import Resize



class Monolayout(pl.LightningModule):
    def __init__(self, learning_rate):
        super().__init__()
        self.encoder = Encoder(num_layers=18, img_ht=128, img_wt=128, pretrained=True)
        self.decoder = Decoder(num_ch_enc=128, num_out=3)
        self.learning_rate = learning_rate
        self.save_hyperparameters()

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y

    def training_step(self, batch, batch_idx):
        rgb, _, pom, _ = batch
        y_hat = self(rgb)
        loss = F.cross_entropy(y_hat, pom.squeeze().long())
        return loss

    def on_validation_start(self) -> None:
        self.img_table = wandb.Table(columns=['ID', 'Image'])
        self.val_iu = np.zeros((3,))
        self.val_ap = np.zeros((3,))
        self.val_counter = 0

    def validation_step(self, batch, batch_idx):
        y_hat = self._shared_eval(batch, batch_idx, "val")

        labels = torch.argmax(y_hat, dim=1).cpu().detach().numpy()
        targets = batch[2].squeeze().long().cpu().detach().numpy()

        for idx in range(y_hat.shape[0]):
            self.val_iu = ((self.val_iu * self.val_counter) + mean_IU(labels[idx], targets[idx], 3))/(self.val_counter+1)
            self.val_ap = ((self.val_ap * self.val_counter) + mean_precision(labels[idx], targets[idx], 3))/(self.val_counter+1)
            self.val_counter += 1


        if batch_idx % 100 == 0:
            X = Resize(labels[0].shape)(batch[0].cpu().detach()).numpy().transpose(0,2,3,1)  # Resize input to shape of label and convert to (B, H, W, C)
            for idx in range(min(4, X.shape[0])):
                mask_img = wandb.Image(X[idx], masks = {
                    "prediction" : {
                        "mask_data" : labels[idx],
                        "class_labels" : {0: 'unknown', 1:'occupied', 2:'free'}
                    },
                    "ground_truth": {
                        "mask_data" : targets[idx],
                        "class_labels" : {0: 'unknown', 1:'occupied', 2:'free'}
                    },
                })
                
                self.img_table.add_data(idx, mask_img)

    def on_validation_end(self) -> None:
        wandb_logger = self.logger.experiment
        wandb_logger.log({"Val. Images" : self.img_table})

        wandb_logger.log({'mIOU/unknown': self.val_iu[0], 'mIOU/occupied': self.val_iu[1], 'mIOU/free': self.val_iu[2]})
        wandb_logger.log({'mAP/unknown': self.val_ap[0], 'mAP/occupied': self.val_ap[1], 'mAP/free': self.val_ap[2]})

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
        self.log(f"{prefix}_loss", loss)
        return y_hat


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)



if __name__ == "__main__":
    model = Monolayout()

    input_rgb = torch.rand((4, 3, 512, 512))
    lbl = torch.randint(3, size=((4, 1, 128, 128))).long()
    model.training_step((input_rgb, None, lbl, None), 0)