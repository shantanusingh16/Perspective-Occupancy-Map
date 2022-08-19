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
    def __init__(self, nblocks=1, nheads=4, head_dim=32, ff_skipcon=True, dropout=0.3, **kwargs):
        super(MultiBlockTransformer, self).__init__()

        self.pos_emb1D = torch.nn.Parameter(torch.randn(1, 128, 64), requires_grad=True)

        self.encoder = Encoder(18, 512, 512, True, last_pool=False)  # B x 128 x H/64 x W/64
        blocks = []
        for _ in range(nblocks):
            blocks.append(MultiheadAttention(128, nheads, head_dim, dropout=dropout)),
            blocks.append(FeedForward(64, 64, skip_conn=ff_skipcon, dropout=dropout)
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
    def __init__(self, learning_rate, *args, **kwargs):
        super().__init__()
        self.model = MultiBlockTransformer(**kwargs)
        self.learning_rate = learning_rate

        self.save_hyperparameters()

    def setup(self, stage= None) -> None:
        if (stage == 'fit') or (stage == 'test') or (stage is None):
            self.eval_metrics = torchmetrics.MetricCollection([torchmetrics.JaccardIndex(3, average=None), \
                torchmetrics.AveragePrecision(3, average=None)])

            ## TODO: Change hard-coded image log frequency to read from config
            self.eval_img_logger = logging.get_genout_logger(self.logger.experiment)


    def forward(self, x):
        y_hat = self.model(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        rgb, _, _, bev, _ = batch
        y_hat = self(rgb)
        y_gt = bev[:,:,64:,32:96]
        loss = F.cross_entropy(y_hat, y_gt)
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        rgb, _, _, bev, _ = batch
        y_hat = self(rgb)
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
        rgb, _, _, _, _ = batch
        y_hat = self(rgb)
        return torch.argmax(y_hat, dim=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)



if __name__ == "__main__":
    model = Indolayout(learning_rate=1e-4)

    input_rgb = torch.rand((4, 3, 512, 512))
    bev = torch.randint(2, size=((4, 3, 128, 128))).float()
    print(bev.unique())

    loss = model.training_step((input_rgb, None, None, bev, None), 0)
    print(loss)