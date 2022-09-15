from typing import Sequence, Tuple
from pytorch_lightning import Callback, LightningModule, Trainer
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional import accuracy

from project.models.utils import MeanSpike

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class OnlineFineTuner(Callback):
    def __init__(
        self, encoder_output_dim: int, num_classes: int, output_all: bool = False
    ) -> None:
        super().__init__()

        self.optimizer: torch.optim.Optimizer

        self.encoder_output_dim = encoder_output_dim
        self.num_classes = num_classes
        self.output_all = output_all

    def on_pretrain_routine_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:

        # add linear_eval layer and optimizer
        if self.output_all is False:
            pl_module.online_finetuner = nn.Linear(
                self.encoder_output_dim, self.num_classes
            ).to(pl_module.device)
        else:
            pl_module.online_finetuner = nn.Sequential(
                MeanSpike(), nn.Linear(self.encoder_output_dim, self.num_classes)
            ).to(pl_module.device)

        self.optimizer = torch.optim.Adam(
            pl_module.online_finetuner.parameters(), lr=1e-4
        )

    def extract_online_finetuning_view(
        self, batch: Sequence
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (finetune_view, _, _), y = batch
        finetune_view = finetune_view.to(device)
        y = y.to(device)

        return finetune_view, y

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        x, y = self.extract_online_finetuning_view(batch)

        with torch.no_grad():
            feats = pl_module(x, mode=pl_module.enc1)

        feats = feats.detach()
        preds = pl_module.online_finetuner(feats)
        loss = F.cross_entropy(preds, y)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        acc = accuracy(F.softmax(preds, dim=1), y)
        pl_module.log("online_train_acc", acc, on_step=True, on_epoch=True)
        pl_module.log("online_train_loss", loss, on_step=True, on_epoch=False)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        x, y = self.extract_online_finetuning_view(batch)

        with torch.no_grad():
            feats = pl_module(x, mode=pl_module.enc1)

        feats = feats.detach()
        preds = pl_module.online_finetuner(feats)
        loss = F.cross_entropy(preds, y)

        acc = accuracy(F.softmax(preds, dim=1), y)
        pl_module.log(
            "online_val_acc",
            acc,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )
        pl_module.log(
            "online_val_loss", loss, on_step=False, on_epoch=True, sync_dist=True
        )
