from argparse import ArgumentParser
from os import times

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
import torchmetrics

from project.models.models import get_model, get_resnet18
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class DVSModule(pl.LightningModule):
    def __init__(self, learning_rate: float, name: str, height: int, width: int, in_channels: int, num_classes: int, pretrained: bool, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=["name", "height", "width", "in_channels", "num_classes"])

        self.height = height
        self.width = width

        if name == 'resnet':
            self.model = get_resnet18(
                in_channels,
                num_classes
            )
        else:
            self.model = get_model(
                name,
                height,
                width,
                in_channels,
                num_classes
            )

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch

        if x.shape[-2] != self.height or x.shape[-1] != self.width:
            x = F.upsample(x, size=(self.height, self.width), mode='nearest').to(device)

        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = torchmetrics.functional.accuracy(y_hat.clone().detach(), y)

        self.log('train_loss', loss, on_epoch=True, prog_bar=False)
        self.log('train_acc', acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        if x.shape[-2] != self.height or x.shape[-1] != self.width:
            x = F.upsample(x, size=(self.height, self.width), mode='nearest').to(device)

        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = torchmetrics.functional.accuracy(y_hat.clone().detach(), y)

        # logs
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch

        if x.shape[-2] != self.height or x.shape[-1] != self.width:
            x = F.upsample(x, size=(self.height, self.width), mode='nearest').to(device)

        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = torchmetrics.functional.accuracy(y_hat.clone().detach(), y)

        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        self.log('test_acc', acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        # Here, you add every arguments needed for your module
        # NOTE: they must appear as arguments in the __init___() function
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        parser.add_argument('--name', type=str, default="resnet")
        parser.add_argument('--pretrained', action="store_true")
        return parser
