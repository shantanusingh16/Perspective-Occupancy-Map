import torch
import torch.nn as nn
from models.multihead_attention import MultiheadAttention
from models.feedforward import FeedForward
from models.encoder_decoder import Encoder, Decoder
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics
from utils import logging


#################################### Basic Transformer ########################################

class P_BasicTransformer(nn.Module):
    def __init__(self):
        super(P_BasicTransformer, self).__init__()

        self.pos_emb1D = torch.nn.Parameter(torch.randn(1, 128, 64), requires_grad=True)

        self.encoder = Encoder(18, 512, 512, True)
        self.basic_transformer = MultiheadAttention(128, 4, 32)
        self.decoder = Decoder(num_ch_enc=128, num_class=3, occ_map_size=64)

        self.scores = None

        self.ft_viz_layer = [self.encoder.conv2]

    def get_attention_map(self):
        return self.scores.mean(dim=1)

    def forward(self, x):
        features = self.encoder(x)
        
        b, c, h, w = features.shape
        features = (features.reshape(b, c, -1) + self.pos_emb1D[:, :, :h*w]).reshape(b, c, h, w)

        features = self.basic_transformer(features, features, features)  # BasicTransformer

        topview = self.decoder(features)
        self.scores = self.basic_transformer.scores

        return topview


#################################### Transformer Multiblock ###################################

class MultiBlockTransformer(nn.Module):
    def __init__(self, nblocks=1):
        super(MultiBlockTransformer, self).__init__()

        self.pos_emb1D = torch.nn.Parameter(torch.randn(1, 128, 64), requires_grad=True)

        self.encoder = Encoder(18, 512, 512, True)
        blocks = []
        for _ in range(nblocks):
            blocks.append(MultiheadAttention(None, 128, 4, 32, dropout=0.3)),
            blocks.append(FeedForward(64, 64, skip_conn=True, dropout=0.3)
        )
        self.transformer = nn.Sequential(*blocks)
        self.decoder = Decoder(num_ch_enc=128, num_class=3, occ_map_size=64)

        self.ft_viz_layer = [blocks[-1]]
        self.scores = []
        
    def get_attention_map(self):
        return self.scores
    
    def forward(self, x):
        features = self.encoder(x)
        
        b, c, h, w = features.shape
        features = (features.reshape(b, c, -1) + self.pos_emb1D[:, :, :h*w]).reshape(b, c, h, w)

        features = self.transformer(features) 
        
        topview = self.decoder(features)

        self.scores = self.transformer._modules['0'].scores

        return topview


#################################### Indolayout Module ########################################


class Indolayout(pl.LightningModule):
    def __init__(self, learning_rate):
        super().__init__()
        self.model = P_BasicTransformer()
        self.learning_rate = learning_rate

        self.save_hyperparameters()

    def setup(self, stage= None) -> None:
        if stage == 'fit' or stage is None:
            self.val_metrics = torchmetrics.MetricCollection([torchmetrics.JaccardIndex(3, average=None), \
                torchmetrics.AveragePrecision(3, average=None)])

            ## TODO: Change hard-coded image log frequency to read from config
            self.val_img_logger = logging.get_genout_logger(self.logger.experiment)


    def forward(self, x):
        y_hat = self.model(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        rgb, _, _, bev = batch
        y_hat = self(rgb)
        y_gt = bev[:,64:,32:96]
        loss = F.cross_entropy(y_hat, y_gt)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self._shared_eval(batch, batch_idx, "val")
        
        y_gt = batch[3][:,64:,32:96]
        rgb = batch[0]

        self.val_metrics.update(y_hat, y_gt)

        y_hat = torch.argmax(y_hat, dim=1).to(torch.uint8) * 127
        y_gt = y_gt.to(torch.uint8) * 127
        self.val_img_logger.log_image(X=rgb, pred=y_hat, gt=y_gt, batch_idx=batch_idx)
       

    def on_validation_epoch_end(self) -> None:
        self.val_img_logger.flush()

        iu, ap =  self.val_metrics.compute().values()
        self.log('mIOU', {'unknown': iu[0], 'occupied': iu[1], 'free': iu[2]}, sync_dist=True)
        self.log('mAP', {'unknown': ap[0], 'occupied': ap[1], 'free': ap[2]}, sync_dist=True)

        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        rgb, _, _, bev = batch
        y_hat = self(rgb)
        y_gt = bev[:,64:,32:96]
        loss = F.cross_entropy(y_hat, y_gt)
        self.log(f"{prefix}_loss", loss, sync_dist=True)
        return y_hat

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        rgb, _, _, _ = batch
        y_hat = self(rgb)
        return torch.argmax(y_hat, dim=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)



if __name__ == "__main__":
    model = Indolayout(learning_rate=1e-4)

    input_rgb = torch.rand((4, 3, 512, 512))
    lbl = torch.randint(3, size=((4, 128, 128))).long()
    loss = model.training_step((input_rgb, None, None, lbl), 0)
    print(loss)