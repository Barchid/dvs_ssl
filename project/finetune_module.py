import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import torchmetrics

from project.models.utils import MeanSpike

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FinetuneModule(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        n_classes: int,
        output_all: bool,
        finetune_all: bool = False,
        mode: str = "snn",
    ):
        super().__init__()
        self.output_all = output_all

        if self.output_all is False:
            self.finetuner = nn.Linear(512, n_classes).to(device)
        else:
            self.finetuner = nn.Sequential(MeanSpike(), nn.Linear(512, n_classes)).to(
                device
            )

        self.finetune_all = finetune_all

        self.encoder = encoder

        self.mode = mode

    def training_step(self, batch, batch_idx):
        (X, Y_a, Y_b), label = batch

        if self.mode == "snn":
            Y_a = Y_a.permute(1, 0, 2, 3, 4)  # from (B,T,C,H,W) to (T, B, C, H, W)

        if self.finetune_all:
            feats = self.encoder(Y_a)
        else:
            with torch.no_grad():
                feats = self.encoder(Y_a)

        preds = self.finetuner(feats)

        loss = F.cross_entropy(preds, label)
        acc = torchmetrics.functional.accuracy(preds, label)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        (X, Y_a, Y_b), label = batch

        if self.mode == "snn":
            X = X.permute(1, 0, 2, 3, 4)  # from (B,T,C,H,W) to (T, B, C, H, W)

        if self.finetune_all:
            feats = self.encoder(X)
        else:
            with torch.no_grad():
                feats = self.encoder(X)

        preds = self.finetuner(feats)

        loss = F.cross_entropy(preds, label)
        acc = torchmetrics.functional.accuracy(preds, label)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)

        return [optimizer], [scheduler]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)  # better perf
