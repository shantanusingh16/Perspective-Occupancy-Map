import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from utils import logging 
from torchvision.transforms import Resize, Normalize, Compose, InterpolationMode
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
import torchmetrics
from models.encoder_decoder import UNetEncoder, UNetDecoder
from models.pon import PON
from torchgeometry.losses import dice_loss


#################################### PON_mod Module ########################################
class PON_mod(PON):
    def __init__(self, learning_rate, num_classes, *args, **kwargs):
        super().__init__(learning_rate, num_classes, *args, **kwargs)
        del self.decoder  # Remove decoder
        self.pon_encoder = UNetEncoder(64, 64)

        self.save_hyperparameters()

    def setup(self, stage= None) -> None:  # Remove setup of logging metrics and instances
        pass

    def forward(self, x):  # Return only bev features and filter
        fts = self.encoder(x)
        proj_fts = [self.dense_transformers[idx](fts[idx]) for idx in range(len(fts))]
        bev_fts, bev_filter = self.cartesian_sample_bev(proj_fts)
        bev_enc_fts = self.pon_encoder(bev_fts)
        return bev_enc_fts, bev_filter




#################################### POMv2 Module ########################################

class POMv2(pl.LightningModule):
    def __init__(self, learning_rate, num_classes, *args, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.output_shape = (128, 128)

        self.sem_pom_model = deeplabv3_mobilenet_v3_large(weights=None, num_classes=num_classes)
        self.sem_pom_model_transforms = Compose([Resize(520), Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        self.sem_pom_output_transform = Resize(self.output_shape, interpolation=InterpolationMode.NEAREST)

        self.setup_projection()

        self.om_encoder = UNetEncoder(3, 64)
        self.om_decoder = UNetDecoder(3, 128)

        self.pon = PON_mod(learning_rate, num_classes)

        self.learning_rate = learning_rate

        self.save_hyperparameters()

    def setup_projection(self):
        h_indices = (torch.arange(self.output_shape[1]) * 2 / self.output_shape[1]) - 1
        w_indices = (torch.arange(self.output_shape[0]) * 2 / self.output_shape[0]) - 1
        h_coords, w_coords = torch.meshgrid((h_indices, w_indices), indexing='ij') 

        img_coords = torch.stack([w_coords, h_coords, torch.ones_like(w_coords)])
        h = 1  # in m, camera_height from floor
        depth = h / (img_coords[1] + 1e-6)  # y/f = h/d
        cam_coords = depth * img_coords

        bev_xsize, bev_zsize = 3.2, 3.2 # m
        bev_res = 0.05 # m
        bev_dim = list(map(int, (bev_zsize/ bev_res, bev_xsize/bev_res)))

        x_indices = ((cam_coords[0] + bev_xsize/2) / bev_res).long()
        z_indices = ((-cam_coords[2] + bev_zsize + 0.1) / bev_res).long()
        grid_filter = (x_indices > 0) & (x_indices < bev_dim[1]) & (z_indices > 0) & (z_indices < bev_dim[0])

        rank = z_indices[grid_filter] * bev_dim[0] + x_indices[grid_filter]
        sort_order = torch.argsort(rank)

        sorted_rank = rank[sort_order]
        rank_filter = torch.cat([sorted_rank[1:] != sorted_rank[:-1], torch.tensor([True], dtype=torch.bool)])
        
        x_indices = x_indices[grid_filter][sort_order][rank_filter]
        z_indices = z_indices[grid_filter][sort_order][rank_filter]

        self.bev_dim = bev_dim
        self.grid_filter = torch.nn.parameter.Parameter(grid_filter, requires_grad=False)
        self.sort_order = torch.nn.parameter.Parameter(sort_order, requires_grad=False)
        self.rank_filter = torch.nn.parameter.Parameter(rank_filter, requires_grad=False)
        self.bev_indices = torch.nn.parameter.Parameter(torch.stack([z_indices, x_indices]), requires_grad=False)

    def project_to_bev(self, pom):
        proj_pom = pom[..., self.grid_filter][..., self.sort_order]

        cumsum_proj_pom = torch.cumsum(proj_pom, dim=-1)[..., self.rank_filter]
        pooled_proj_pom = torch.cat([cumsum_proj_pom[..., :1], cumsum_proj_pom[...,1:] - cumsum_proj_pom[...,:-1]], dim=-1)

        cumsum_rank_count = torch.cumsum(torch.ones_like(proj_pom), dim=-1)[..., self.rank_filter]
        rank_count = torch.cat([cumsum_rank_count[..., :1], cumsum_rank_count[...,1:] - cumsum_rank_count[...,:-1]], dim=-1)

        avg_proj_pom = pooled_proj_pom / rank_count

        bev = torch.zeros((pom.shape[0], self.num_classes, *self.bev_dim), requires_grad=False, dtype=torch.float32, device=pom.device)
        bev[..., self.bev_indices[0], self.bev_indices[1]] = avg_proj_pom

        return bev
 
    def setup(self, stage= None) -> None:
        if (stage == 'fit') or (stage == 'test') or (stage is None):
            self.eval_pom_metrics = torchmetrics.MetricCollection([torchmetrics.JaccardIndex(self.num_classes, average=None), \
                torchmetrics.AveragePrecision(self.num_classes, average=None)])
            self.eval_bev_metrics = torchmetrics.MetricCollection([torchmetrics.JaccardIndex(self.num_classes, average=None), \
                torchmetrics.AveragePrecision(self.num_classes, average=None)])

            ## TODO: Change hard-coded image log frequency to read from config
            self.eval_pom_logger = logging.get_segout_logger(self.logger.experiment)
            self.eval_bev_logger = logging.get_genout_logger(self.logger.experiment)

    def forward(self, x):
        pon_bev_fts, pon_bev_filter = self.pon(x)

        x = self.sem_pom_model_transforms(x)
        y_hat = self.sem_pom_model(x)['out']
        pom_hat = self.sem_pom_output_transform(y_hat)

        proj_bev = self.project_to_bev(F.softmax(pom_hat, dim=1))
        proj_bev_fts = self.om_encoder(proj_bev)

        combined_bev_fts = {}
        for k in proj_bev_fts.keys():
            combined_bev_fts[k] = torch.cat([proj_bev_fts[k], pon_bev_fts[k]], dim=1)

        bev_hat = self.om_decoder(combined_bev_fts)

        return pom_hat, bev_hat

    def training_step(self, batch, batch_idx):
        rgb, _, pom, bev, _ = batch
        pom_hat, bev_hat = self(rgb)

        bev = bev[:,:,64:,32:96]
        # loss = F.cross_entropy(pom_hat, pom) + F.cross_entropy(bev_hat, bev)
        loss = dice_loss(bev_hat, torch.argmax(bev, dim=1)) + dice_loss(pom_hat, torch.argmax(pom, dim=1))
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, stage):
        rgb, _, pom, bev, _ = batch

        pom_hat, bev_hat = self(rgb)
        bev = bev[:,:,64:,32:96]

        # loss_pom = F.cross_entropy(pom_hat, pom)
        # loss_bev = F.cross_entropy(bev_hat, bev)
        # loss = loss_pom + loss_bev

        loss_pom = dice_loss(pom_hat, torch.argmax(pom, dim=1))
        loss_bev = dice_loss(bev_hat, torch.argmax(bev, dim=1))
        loss = loss_pom + loss_bev

        self.log(f"{stage}_loss", loss, sync_dist=True)
        self.log(f"pom/{stage}/loss", loss_pom, sync_dist=True)
        self.log(f"bev/{stage}/loss", loss_bev, sync_dist=True)

        pom = torch.argmax(pom, dim=1)
        bev = torch.argmax(bev, dim=1)
        self.eval_pom_metrics.update(pom_hat, pom)
        self.eval_bev_metrics.update(bev_hat, bev)

        pom_hat = torch.argmax(pom_hat, dim=1).to(torch.uint8) * (255 // (self.num_classes - 1))
        bev_hat = torch.argmax(bev_hat, dim=1).to(torch.uint8) * (255 // (self.num_classes - 1))
        pom = pom.to(torch.uint8) * (255 // (self.num_classes - 1))
        bev = bev.to(torch.uint8) * (255 // (self.num_classes - 1))

        self.eval_pom_logger.log_image(X=rgb, pred=pom_hat, gt=pom, batch_idx=batch_idx)
        self.eval_bev_logger.log_image(X=rgb, pred=bev_hat, gt=bev, batch_idx=batch_idx)

    def on_validation_epoch_end(self) -> None:
        self.log_metrics_and_outputs(stage='val')

    def on_test_epoch_end(self) -> None:
        self.log_metrics_and_outputs(stage='test')

    def log_metrics_and_outputs(self, stage):
        self.eval_pom_logger.flush(stage)
        self.eval_bev_logger.flush(stage)

        iu, ap =  self.eval_pom_metrics.compute().values()
        self.log(f'pom/{stage}/mIOU', {'unknown': iu[0], 'occupied': iu[1], 'free': iu[2]}, sync_dist=True)
        self.log(f'pom/{stage}/mAP', {'unknown': ap[0], 'occupied': ap[1], 'free': ap[2]}, sync_dist=True)

        iu, ap =  self.eval_bev_metrics.compute().values()
        self.log(f'bev/{stage}/mIOU', {'unknown': iu[0], 'occupied': iu[1], 'free': iu[2]}, sync_dist=True)
        self.log(f'bev/{stage}/mAP', {'unknown': ap[0], 'occupied': ap[1], 'free': ap[2]}, sync_dist=True)

        self.eval_pom_metrics.reset()
        self.eval_bev_metrics.reset()

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        rgb, _, _, _, _ = batch
        pom_hat, bev_hat = self(rgb)
        return torch.argmax(pom_hat, dim=1), torch.argmax(bev_hat, dim=1)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        shd = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10, threshold_mode='rel', threshold=1e-3)

        return {'optimizer':opt, 'lr_scheduler':shd, 'monitor': 'val_loss'}


if __name__ == "__main__":
    model = POMv2(learning_rate=1e-3, num_classes=3)

    input_rgb = torch.rand((4, 3, 512, 512))
    pom = F.softmax(torch.rand((4, 3, 128, 128)), dim=1)
    bev = F.softmax(torch.rand((4, 3, 128, 128)), dim=1)
    
    loss = model.training_step((input_rgb, None, pom, bev, None), 0)
    print(loss)