from argparse import ArgumentParser
from os import times

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from spikingjelly.clock_driven import functional
import pytorch_lightning as pl
import torchmetrics

from project.models.models import get_encoder
from project.models.snn_models import get_encoder_snn
from project.models.utils import MeanSpike


class ClassifModule(pl.LightningModule):
    def __init__(
        self,
        n_classes: int,
        learning_rate: float,
        epochs: int,
        timesteps: int,
        mode: str = "cnn",
        output_all: bool = False,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["epochs", "n_classes", "timesteps"])
        self.output_all = output_all
        self.mode = mode
        self.epochs = epochs

        if mode == "cnn":
            self.encoder = get_encoder(in_channels=2 * timesteps)
        elif mode == "snn":
            self.encoder = get_encoder_snn(2, timesteps, output_all=output_all)

        if self.mode == "snn" and output_all is True:
            self.fc = nn.Sequential(MeanSpike(), nn.Linear(512, n_classes))
        else:
            self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        if self.mode == "snn":
            functional.reset_net(self.encoder)
            x = x.permute(1, 0, 2, 3, 4)  # from (B,T,C,H,W) to (T, B, C, H, W)

        x = self.model(x)
        x = self.fc(x)
        return x

    def shared_step(self, batch):
        (X, Y_a, Y_b), label = batch

        y_hat = self(Y_a)
        loss = F.cross_entropy(y_hat, label)
        acc = torchmetrics.functional.accuracy(y_hat.clone().detach(), label)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)

        # logs
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

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