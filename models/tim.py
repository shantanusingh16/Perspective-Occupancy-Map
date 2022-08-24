import torch
import torch.nn as nn
from models.encoder_decoder import double_conv, FPN
from models.transformers import Transformer1D, PositionalEncoding
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics
from utils import logging
from einops import rearrange, repeat


#################################### DecodeAndMap Module ########################################

class DecodeAndMap(nn.Module):
    def __init__(self, in_dim, out_dim, nheads, head_dim, mha_skipcon, dropout, ff_dim, ff_skipcon) -> None:
        super().__init__()
        self.transformer = Transformer1D(in_dim, nheads, head_dim, mha_skipcon, dropout, ff_dim, ff_skipcon)
        self.fc =  nn.Linear(in_dim, out_dim)

    def forward(self, x_key, x_query=None, x_value=None):
        y_hat = self.transformer(x_key, x_query, x_value)
        y_hat = self.fc(y_hat)
        return y_hat



#################################### TIM Module ########################################


class TIM(pl.LightningModule):
    def __init__(self, learning_rate, num_classes, *args, **kwargs):
        super().__init__()
        self.encoder = FPN()
        self.bev_ch = 128
        self.setup_mem_encoders(4, 64, 0.3, 64, True)
        self.setup_bev_decoders(4, 64, 0.3, 64, True)
        self.decoder = nn.Sequential(double_conv(self.bev_ch, 64), double_conv(64, 32), double_conv(32, 16), \
            double_conv(16, 8), double_conv(8, num_classes))

        self.learning_rate = learning_rate
        self.num_classes = num_classes

        self.save_hyperparameters()

    def setup_mem_encoders(self, nheads, head_dim, dropout, ff_dim, ff_skipcon):
        self.input_shape = torch.tensor([512, 512], dtype=int)

        mem_encoders = []  # Self-attention encoders
        mem_col_pe = []
        mem_ang_pe = []
        for idx, s in enumerate(self.encoder.scales):
            W, H = torch.div(self.input_shape, s, rounding_mode='trunc').long()
            C = self.encoder.num_channels[idx]
            mem_encoders.append(Transformer1D(C, nheads, head_dim, True, dropout, ff_dim, ff_skipcon))
            mem_col_pe.append(PositionalEncoding(C, H))
            mem_ang_pe.append(PositionalEncoding(C*H, W))

        self.mem_encoders = nn.ModuleList(mem_encoders)
        self.mem_col_pe = nn.ModuleList(mem_col_pe)
        self.mem_ang_pe = nn.ModuleList(mem_ang_pe)

    def setup_bev_decoders(self, nheads, head_dim, dropout, ff_dim, ff_skipcon):
        self.focal_length = 256  
        self.bev_shape = torch.tensor([64, 64], dtype=int)
        self.bev_res = 0.05 # in m

        self.z_dist_thresh = torch.tensor([3.2] + [(self.focal_length * self.bev_res / s) for s in self.encoder.scales])
        self.x_dist_thresh = []
        self.bevdec_z = []  # BEV Decoder Channels

        bev_decoders = [] # Cross-attention decoders
        bev_ray_pe = []
        bev_ang_pe = []
        bev_rayq = []

        for idx, s in enumerate(self.encoder.scales):
            W, H = torch.div(self.input_shape, s, rounding_mode='trunc').long()
            C = self.encoder.num_channels[idx]
            Z = torch.ceil((self.z_dist_thresh[idx] - self.z_dist_thresh[idx + 1]) / self.bev_res).long()
            self.x_dist_thresh.append(W * self.bev_res / 2)
            self.bevdec_z.append(Z)
            
            bev_rayq.append(nn.parameter.Parameter(torch.rand((Z, C) ,dtype=torch.float32), requires_grad=True))
            bev_decoders.append(DecodeAndMap(C, self.bev_ch, nheads, head_dim, False, dropout, ff_dim, ff_skipcon))
            bev_ray_pe.append(PositionalEncoding(C, Z))
            bev_ang_pe.append(PositionalEncoding(C*Z, self.bev_shape[1]))

        self.x_dist_thresh = torch.tensor(self.x_dist_thresh, dtype=torch.float32, requires_grad=False)
        self.bevdec_z = torch.tensor(self.bevdec_z, dtype=torch.long, requires_grad=False)

        self.bev_decoders = nn.ModuleList(bev_decoders)
        self.bev_ray_pe = nn.ModuleList(bev_ray_pe)
        self.bev_ang_pe = nn.ModuleList(bev_ang_pe)
        self.bev_rayq = nn.ParameterList(bev_rayq)
        
    def cartesian_sample_bev(self, proj_fts):
        B = proj_fts[0].shape[0]
        bev = torch.zeros((B, self.bev_ch, *self.bev_shape), requires_grad=False, dtype=torch.float32, device=proj_fts[0].device)
        h_indices, w_indices = torch.meshgrid([torch.arange(self.bev_shape[1]), torch.arange(self.bev_shape[0])], indexing='ij')

        z_vals = (-h_indices + self.bev_shape[1]) * self.bev_res
        x_vals = (w_indices - (self.bev_shape[0]/2)) * self.bev_res

        proj_ft_indices = torch.floor(torch.log2(z_vals/0.1)).long()
        proj_ft_indices = len(proj_fts) - 1 - torch.clamp(proj_ft_indices, 0, len(proj_fts) - 1)

        z_ft_indices = self.bevdec_z[proj_ft_indices] - torch.floor((self.z_dist_thresh[proj_ft_indices] - z_vals)/self.bev_res).long()

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
        B = x.shape[0]
        fpn_fts = self.encoder(x)

        # Add Column Position Encoding to FPN Extracted features
        fts_flat = [rearrange(ft, 'b c h w -> (b w) h c') for ft in fpn_fts]        
        fts_flat = [self.mem_col_pe[idx](ft) for idx, ft in enumerate(fts_flat)]

        # Generate Memory Embeddings
        mem_enc_fts = [self.mem_encoders[idx](ft) for idx, ft in enumerate(fts_flat)]
        
        # Add Angular Position Encoding to Memory Embeddings
        mem_enc_fts = [rearrange(ft, '(b w) h c -> b w (h c)', b=B) for ft in mem_enc_fts]        
        mem_enc_fts = [self.mem_ang_pe[idx](ft) for idx, ft in enumerate(mem_enc_fts)]  
        mem_enc_fts = [rearrange(ft, 'b w (h c) -> (b w) h c', c=fpn_fts[idx].shape[1]) for idx, ft in enumerate(mem_enc_fts)]   

        # Generate BEV queries
        bev_rayq = [repeat(rayq, 'z c -> (b w) z c', b=B, w=fpn_fts[idx].shape[3]) for idx, rayq in enumerate(self.bev_rayq)]  # qk.shape = (B, Zk, Ck)

        # Add Ray Position Encoding to BEV Queries
        bev_rayq = [self.bev_ray_pe[idx](ft) for idx, ft in enumerate(bev_rayq)] # qk.shape = (B*Wk, Zk, Ck)

        # Add Angular Position Encoding to BEV queries
        bev_rayq = [rearrange(ft, '(b w) z c -> b w (z c)', b=B) for ft in bev_rayq] 
        bev_rayq = [self.bev_ang_pe[idx](ft) for idx, ft in enumerate(bev_rayq)]
        bev_rayq = [rearrange(ft, 'b w (z c) -> (b w) z c', c=fpn_fts[idx].shape[1]) for idx, ft in enumerate(bev_rayq)]   

        # Decode Polar BEV Features
        bev_ray_fts = [self.bev_decoders[idx](mem_enc_fts[idx], bev_rayq[idx]) for idx in range(len(mem_enc_fts))]
        bev_ray_fts = [rearrange(ft, '(b w) z c -> b c z w', b=B) for ft in bev_ray_fts] 

        # Sample to Grid BEV Features
        bev_fts, bev_filter = self.cartesian_sample_bev(bev_ray_fts)
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
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        shd = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10, threshold_mode='rel', threshold=1e-3)

        return {'optimizer':opt, 'lr_scheduler':shd, 'monitor': 'val_loss'}



if __name__ == "__main__":
    model = TIM(learning_rate=1e-4, num_classes=3)
    model.cuda()

    input_rgb = torch.rand((4, 3, 512, 512)).cuda()
    bev = F.softmax(torch.rand((4, 3, 128, 128)), dim=1).cuda()

    loss = model.training_step((input_rgb, None, None, bev, None), 0)
    print(loss)