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
        self,
        encoder_output_dim: int,
        num_classes: int,
        output_all: bool = False,
        enc="enc1",
    ) -> None:
        super().__init__()

        self.optimizer: torch.optim.Optimizer

        self.encoder_output_dim = encoder_output_dim
        self.num_classes = num_classes
        self.output_all = output_all
        self.enc = enc

    def on_pretrain_routine_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:

        if self.enc == "enc2":
            pl_module.online_finetuner2 = nn.Linear(
                self.encoder_output_dim, self.num_classes
            ).to(pl_module.device)

            self.optimizer = torch.optim.Adam(
                pl_module.online_finetuner2.parameters(), lr=1e-4
            )
        else:
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
            if self.enc == "enc2":
                feats = pl_module(x, mode=pl_module.enc2, enc=2)
            else:
                feats = pl_module(x, mode=pl_module.enc1)

        feats = feats.detach()
        if self.enc == "enc2":
            preds = pl_module.online_finetuner2(feats)
        else:
            preds = pl_module.online_finetuner(feats)

        loss = F.cross_entropy(preds, y)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        acc = accuracy(F.softmax(preds, dim=1), y)
        pl_module.log(
            f"online_train_acc_{self.enc}",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        pl_module.log(
            f"online_train_loss_{self.enc}", loss, on_step=False, on_epoch=True
        )

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
            if self.enc == "enc2":
                feats = pl_module(x, mode=pl_module.enc2, enc=2)
            else:
                feats = pl_module(x, mode=pl_module.enc1)

        feats = feats.detach()
        if self.enc == "enc2":
            preds = pl_module.online_finetuner2(feats)
        else:
            preds = pl_module.online_finetuner(feats)

        loss = F.cross_entropy(preds, y)

        acc = accuracy(F.softmax(preds, dim=1), y)
        pl_module.log(
            f"online_val_acc_{self.enc}",
            acc,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )
        pl_module.log(
            f"online_val_loss_{self.enc}",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
