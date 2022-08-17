import wandb
import torch
from torchvision.transforms.functional import resize   


class SegOutLogger():

    def __init__(self, logger, log_frequency=4, max_count=200) -> None:
        self.logger = logger
        self.data = []
        self.log_frequency = log_frequency
        self.max_count = max_count

    def log_image(self, X, pred, gt, batch_idx):
        pred = torch.argmax(pred, dim=1).cpu().detach().numpy()   # (B,C,H,W) -> (B,H,W)
        gt = gt.cpu().detach().numpy()

        X = resize(X.cpu().detach(), pred[0].shape) # Resize input to shape of pred
        X = X.numpy().transpose(0,2,3,1)   # (B,C,H,W) -> (B,H,W,C)
        for idx in range(0, X.shape[0], self.log_frequency):
            self.data.append((batch_idx + idx, X[idx], pred[idx], gt[idx]))  # (idx, rgb, pred, gt)

    def flush(self):
        img_table = wandb.Table(columns=['ID', 'Image'])
        for item in self.data:
            mask_img = wandb.Image(item[1], masks = {
                "prediction" : {
                    "mask_data" : item[2],
                    "class_labels" : {0: 'unknown', 1:'occupied', 2:'free'}
                },
                "ground_truth": {
                    "mask_data" : item[3],
                    "class_labels" : {0: 'unknown', 1:'occupied', 2:'free'}
                },
            })
            
            img_table.add_data(item[0], mask_img)

        self.logger.log({"Val. Segmentation Results" : img_table})
        self.data = []



class GenOutLogger():

    def __init__(self, logger, log_frequency=4, max_count=200) -> None:
        self.logger = logger
        self.data = []
        self.log_frequency = log_frequency
        self.max_count = max_count

    def log_image(self, X, pred, gt, batch_idx):
        pred = pred.cpu().detach().numpy()   # (B,C,H,W) -> (B,H,W)
        gt = gt.cpu().detach().numpy()

        X = resize(X.cpu().detach(), pred[0].shape) # Resize input to shape of pred
        X = X.numpy().transpose(0,2,3,1)   # (B,C,H,W) -> (B,H,W,C)
        for idx in range(0, X.shape[0], self.log_frequency):
            self.data.append((batch_idx + idx, X[idx], pred[idx], gt[idx]))  # (idx, rgb, pred, gt)

    def flush(self):
        img_table = wandb.Table(columns=['ID', 'Input', 'Pred', 'GT'])
        for item in self.data:            
            img_table.add_data(item[0], wandb.Image(item[1]), wandb.Image(item[2]), wandb.Image(item[3]))

        self.logger.log({"Val. Predictions" : img_table})
        self.data = []