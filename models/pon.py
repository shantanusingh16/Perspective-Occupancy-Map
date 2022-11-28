import numpy as np
import torch
import torch.nn as nn
from models.encoder_decoder import double_conv, FPN
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics
from utils import logging


#################################### DenseTransformer Block ########################################

class DenseTransformer(nn.Module):
    def __init__(self, C, H, W, Z, bottleneck=128, out_ch=64) -> None:
        super(DenseTransformer, self).__init__()
        self.in_shape = (C, H, W)  
        self.bottleneck = bottleneck
        self.out_ch = out_ch
        self.fc1 = nn.Linear(C*H, self.bottleneck)
        self.fc2 = nn.Linear(self.bottleneck, self.out_ch*Z)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.transpose(1, 3)
        x = x.reshape((B, W, C*H))
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.reshape((B, W, -1, self.out_ch))
        x = x.transpose(1, 3)
        return x



#################################### PON Module ########################################


class PON(pl.LightningModule):
    def __init__(self, learning_rate, num_classes, *args, **kwargs):
        super().__init__()
        self.encoder = FPN()
        self.setup_dense_transformers(bottleneck=128, out_ch=64)
        self.decoder = nn.Sequential(double_conv(64, 32), double_conv(32, 16), \
            double_conv(16, 8), double_conv(8, num_classes))

        self.learning_rate = learning_rate
        self.num_classes = num_classes

        self.save_hyperparameters()

    def setup_dense_transformers(self, bottleneck, out_ch):
        self.input_shape = torch.tensor([512, 512], dtype=int)
        self.focal_length = 256  
        self.bev_shape = torch.tensor([64, 64], dtype=int)
        self.bev_res = 0.05 # in m

        self.z_dist_thresh = torch.tensor([3.2] + [(self.focal_length * self.bev_res / s) for s in self.encoder.scales])
        self.x_dist_thresh = []
        self.dt_ch = []

        dense_transformers = []
        for idx, s in enumerate(self.encoder.scales):
            W, H = torch.div(self.input_shape, s, rounding_mode='trunc').long()
            C = self.encoder.num_channels[idx]
            Z = torch.ceil((self.z_dist_thresh[idx] - self.z_dist_thresh[idx + 1]) / self.bev_res).long()
            dense_transformers.append(DenseTransformer(C, H, W, Z, bottleneck, out_ch))

            self.x_dist_thresh.append(W * self.bev_res / 2)
            self.dt_ch.append(Z)

        self.x_dist_thresh = torch.tensor(self.x_dist_thresh, dtype=torch.float32, requires_grad=False)
        self.dt_ch = torch.tensor(self.dt_ch, dtype=torch.long, requires_grad=False)
        self.dense_transformers = nn.ModuleList(dense_transformers)
        

    def cartesian_sample_bev(self, proj_fts):
        B, C, _, _ = proj_fts[0].shape
        bev = torch.zeros((B, C, *self.bev_shape), requires_grad=False, dtype=torch.float32, device=proj_fts[0].device)
        h_indices, w_indices = torch.meshgrid([torch.arange(self.bev_shape[1]), torch.arange(self.bev_shape[0])], indexing='ij')

        z_vals = (-h_indices + self.bev_shape[1]) * self.bev_res
        x_vals = (w_indices - (self.bev_shape[0]/2)) * self.bev_res

        proj_ft_indices = torch.floor(torch.log2(z_vals/0.1)).long()
        proj_ft_indices = len(proj_fts) - 1 - torch.clamp(proj_ft_indices, 0, len(proj_fts) - 1)

        z_ft_indices = self.dt_ch[proj_ft_indices] - torch.floor((self.z_dist_thresh[proj_ft_indices] - z_vals)/self.bev_res).long()

        grid_filter = abs(x_vals) <= self.x_dist_thresh[proj_ft_indices]

        ft_scales = torch.tensor(self.encoder.scales)[proj_ft_indices]
        u_vals = (x_vals * self.focal_length) / (ft_scales * z_vals)
        u_vals = (u_vals + (self.input_shape[0] / (2 * ft_scales))).long()

        for idx, proj_ft in enumerate(proj_fts):
            filter_idx = (proj_ft_indices == idx) & grid_filter

            flt_z_ft_indices = torch.clamp(z_ft_indices[filter_idx], 0, proj_ft.shape[2] - 1)
            flt_u_vals = torch.clamp(u_vals[filter_idx], 0, proj_ft.shape[3] - 1)

            flt_hidx = h_indices[filter_idx]
            flt_widx = w_indices[filter_idx]

            bev[:, :, flt_hidx, flt_widx] = proj_ft[:, :, flt_z_ft_indices, flt_u_vals]
        
        return bev, grid_filter


    def setup(self, stage= None) -> None:
        if (stage == 'fit') or (stage == 'test') or (stage is None):
            self.eval_metrics = torchmetrics.MetricCollection([torchmetrics.JaccardIndex(3, average=None), \
                torchmetrics.AveragePrecision(3, average=None)])

            ## TODO: Change hard-coded image log frequency to read from config
            self.eval_img_logger = logging.get_genout_logger(self.logger.experiment)


    def forward(self, x):
        fts = self.encoder(x)
        proj_fts = [self.dense_transformers[idx](fts[idx]) for idx in range(len(fts))]
        bev_fts, bev_filter = self.cartesian_sample_bev(proj_fts)
        bev_hat = self.decoder(bev_fts)
        return bev_hat, bev_filter

    def training_step(self, batch, batch_idx):
        rgb, _, _, bev, _ = batch
        y_hat, y_mask = self(rgb)
        y_gt = bev[:,:,64:,32:96]
        loss = F.cross_entropy(y_hat, y_gt)
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        rgb, _, _, bev, _ = batch
        y_hat, y_mask = self(rgb)
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
        y_hat, y_mask = self(rgb)
        return torch.argmax(y_hat, dim=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)



if __name__ == "__main__":
    model = PON(learning_rate=1e-4, num_classes=3)
    model.cuda()

    input_rgb = torch.rand((4, 3, 512, 512)).cuda()
    bev = F.softmax(torch.rand((4, 3, 128, 128)), dim=1).cuda()

    loss = model.training_step((input_rgb, None, None, bev, None), 0)
    print(loss)