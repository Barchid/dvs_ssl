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
from utils.forward_analyze import forward_analyze_3dcnn, forward_analyze_cnn, forward_analyze_snn


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
        elif mode == "snn2":
            self.encoder = get_encoder_snn_2(2, timesteps, output_all=output_all)
        elif mode == "3dcnn":
            self.encoder = get_encoder_3d(in_channels=2)

        if self.mode == "snn" and output_all is True:
            self.fc = nn.Sequential(MeanSpike(), nn.Linear(512, n_classes))
        else:
            self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        if self.mode == "snn" or self.mode == "snn2":
            functional.reset_net(self.encoder)
            x = x.permute(1, 0, 2, 3, 4)  # from (B,T,C,H,W) to (T, B, C, H, W)

        if self.mode == "3dcnn":
            x = x.permute(0, 2, 1, 3, 4)  # from (B,T,C,H,W) to (B,C,T,H,W)

        x = self.encoder(x)
        x = self.fc(x)
        return x
    
    def forward_analyze(self, x):
        if self.mode == "snn" or self.mode == "snn2":
            functional.reset_net(self.encoder)
            x = x.permute(1, 0, 2, 3, 4)  # from (B,T,C,H,W) to (T, B, C, H, W)

        if self.mode == "3dcnn":
            x = x.permute(0, 2, 1, 3, 4)  # from (B,T,C,H,W) to (B,C,T,H,W)

        if self.mode == "3dcnn":
            feats, stem_feat, res2_feat, res3_feat, res4_feat = forward_analyze_3dcnn(self.encoder, x)
        elif self.mode == "cnn":
            feats, stem_feat, res2_feat, res3_feat, res4_feat = forward_analyze_cnn(self.encoder, x)
        else: # mode == "snn"
            feats, stem_feat, res2_feat, res3_feat, res4_feat = forward_analyze_snn(self.encoder, x)
        
        feats = self.encoder(x)
        x = self.fc(feats)        
        return x, feats.flatten(), stem_feat.flatten(1), res2_feat.flatten(1), res3_feat.flatten(1), res4_feat.flatten(1)

    def shared_step(self, x, label):
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, label)
        acc = torchmetrics.functional.accuracy(y_hat.clone().detach(), label)
        return loss, acc

    def training_step(self, batch, batch_idx):
        (X, Y_a, Y_b), label = batch
        loss, acc = self.shared_step(Y_a, label)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (X, Y_a, Y_b), label = batch
        loss, acc = self.shared_step(X, label)

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
