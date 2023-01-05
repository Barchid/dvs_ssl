from argparse import ArgumentParser
from os import times

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from spikingjelly.clock_driven import functional
import pytorch_lightning as pl
import torchmetrics

from project.models.models import get_encoder, get_encoder_3d
from project.models.snn_models import get_encoder_snn, get_encoder_snn_2
from project.models.utils import MeanSpike


class LocalizationModule(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float,
        epochs: int,
        encoder: nn.Module,
        mode: str = "cnn",
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["epochs"])
        self.mode = mode
        self.epochs = epochs

        self.encoder = encoder
        self.fc = nn.Linear(512, 4)
        
        self.criterion = DIoULoss()

    def forward(self, x):
        if self.mode == "snn":
            functional.reset_net(self.encoder)
            x = x.permute(1, 0, 2, 3, 4)  # from (B,T,C,H,W) to (T, B, C, H, W)

        if self.mode == "3dcnn":
            x = x.permute(0, 2, 1, 3, 4)  # from (B,T,C,H,W) to (B,C,T,H,W)

        x = self.encoder(x)
        x = self.fc(x)
        return x

    def shared_step(self, x, bbox):
        y_hat = self(x)
        loss = self.criterion(y_hat, bbox)
        iou = compute_IoU(y_hat.clone().detach(), bbox)
        return loss, iou

    def training_step(self, batch, batch_idx):
        X, bbox = batch
        loss, iou = self.shared_step(X, bbox)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_iou", iou, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, bbox = batch
        loss, iou = self.shared_step(X, bbox)

        # logs
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_iou", iou, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)

        return [optimizer], [scheduler]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)  # better perf


class DIoULoss(nn.Module):
    """Distance-IoU loss function for single object localization"""

    def __init__(self):
        super(DIoULoss, self).__init__()

    def forward(self, pred, gt):
        x1 = pred[:, 0]
        y1 = pred[:, 1]
        x2 = pred[:, 2]
        y2 = pred[:, 3]

        x1g = gt[:, 0]
        y1g = gt[:, 1]
        x2g = gt[:, 2]
        y2g = gt[:, 3]

        x2 = torch.max(x1, x2)
        y2 = torch.max(y1, y2)

        x_p = (x2 + x1) / 2
        y_p = (y2 + y1) / 2
        x_g = (x1g + x2g) / 2
        y_g = (y1g + y2g) / 2

        xkis1 = torch.max(x1, x1g)
        ykis1 = torch.max(y1, y1g)
        xkis2 = torch.min(x2, x2g)
        ykis2 = torch.min(y2, y2g)

        xc1 = torch.min(x1, x1g)
        yc1 = torch.min(y1, y1g)
        xc2 = torch.max(x2, x2g)
        yc2 = torch.max(y2, y2g)

        intsctk = torch.zeros(x1.size()).to(pred)
        mask = (ykis2 > ykis1) * (xkis2 > xkis1)
        intsctk[mask] = (xkis2[mask] - xkis1[mask]) * \
            (ykis2[mask] - ykis1[mask])
        unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * \
            (y2g - y1g) - intsctk + 1e-7
        iouk = intsctk / unionk

        c = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) + 1e-7
        d = ((x_p - x_g) ** 2) + ((y_p - y_g) ** 2)
        u = d / c
        diouk = iouk - u
        diouk = (1 - diouk).sum(0) / pred.size(0)

        return diouk


def compute_IoU(pred, gt):
    x1 = pred[:, 0]
    y1 = pred[:, 1]
    x2 = pred[:, 2]
    y2 = pred[:, 3]

    x1g = gt[:, 0]
    y1g = gt[:, 1]
    x2g = gt[:, 2]
    y2g = gt[:, 3]

    x2 = torch.max(x1, x2)
    y2 = torch.max(y1, y2)

    x_p = (x2 + x1) / 2
    y_p = (y2 + y1) / 2
    x_g = (x1g + x2g) / 2
    y_g = (y1g + y2g) / 2

    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    intsctk = torch.zeros(x1.size()).to(pred)
    mask = (ykis2 > ykis1) * (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * \
        (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * \
        (y2g - y1g) - intsctk + 1e-7
    iouk = intsctk / unionk

    c = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) + 1e-7
    d = ((x_p - x_g) ** 2) + ((y_p - y_g) ** 2)
    u = d / c
    iouk = iouk.sum(0) / pred.size(0)

    return iouk