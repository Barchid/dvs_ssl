from project.losses.barlow_twins_loss import BarlowTwinsLoss
from project.losses.vicreg_loss import VICRegLoss
from project.models.models import get_encoder, get_projector
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.functional import accuracy

from project.models.snn_models import get_encoder_snn, get_projector_liaf
from project.models.utils import MeanSpike

class SSLModule(pl.LightningModule):
    def __init__(self, n_classes: int, learning_rate: float, epochs: int, timesteps: int, ssl_loss: str = 'barlow_twins', enc1: str = 'cnn', enc2: str = 'cnn', proj1: str = 'ann', proj2: str = 'ann', **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['epochs', 'n_classes', 'ssl_loss', 'timesteps'])
        self.epochs = epochs
        self.enc1 = enc1
        self.enc2 = enc2
        self.proj1 = proj1
        self.proj2 = proj2
        
        self.encoder = None
        self.projector = None
        self.encoder1 = None
        self.encoder2 = None
        self.projector1 = None
        self.projector2 = None
        
        # this is ugly af but nvm
        if proj1 == proj2:
            if proj1 == 'ann':
                self.projector = get_projector()
            else:
                self.projector = get_projector_liaf()
        else:
            if proj1 == 'ann':
                self.projector1 = get_projector()
            else:
                self.projector1 = get_projector_liaf()

            if proj2 == 'ann':
                self.projector2 = get_projector()
            else:
                self.projector2 = get_projector_liaf()
        
        if enc1 == enc2:
            if enc1 == 'cnn':
                self.encoder = get_encoder(in_channels=2 * timesteps)
            elif enc1 == 'snn':
                self.encoder = get_encoder_snn(2, timesteps, output_all=proj1 != 'ann')
        else:
            if enc1 == 'cnn':
                self.encoder1 = get_encoder(in_channels=2 * timesteps)
            elif enc1 == 'snn':
                self.encoder1 = get_encoder_snn(2, timesteps, output_all=proj1 != 'ann')
                
            if enc2 == 'cnn':
                self.encoder2 = get_encoder(in_channels=2 * timesteps)
            elif enc2 == 'snn':
                self.encoder2 = get_encoder_snn(2, timesteps, output_all=proj2 != 'ann')
            
        # either barlow twins or VICReg
        if ssl_loss == 'barlow_twins':
            self.criterion = BarlowTwinsLoss()
        else:
            self.criterion = VICRegLoss()
        
        # Attributes used to evaluate the feature extractor
        if proj1 == 'ann':
            self.eval_fc = nn.Linear(512, n_classes) # Linear layer used to evaluate our model (512 for ResNet-18)
        else:
            self.eval_fc = nn.Sequential([
                MeanSpike(),
                nn.Linear(512, n_classes)
            ])
            
        self.eval_optimizer = torch.optim.Adam(self.eval_fc.parameters(), lr=1e-3)

    def forward(self, Y, enc=None):
        if enc is None:    
            representation = self.encoder(Y)
            Z = self.projector(representation)
        elif enc == 1:
            representation = self.encoder1(Y)
            Z = self.projector1(representation)
        else:
            representation = self.encoder2(Y)
            Z = self.projector2(representation)
            
        return representation, Z
    
    def shared_step(self, batch):
        (X, Y_a, Y_b), label = batch
        
        if self.encoder is None:
            _, Z_a = self(Y_a, enc=1)
            _, Z_b = self(Y_b, enc=2)
        else:
            _, Z_a = self(Y_a)
            _, Z_b = self(Y_b)
        loss = self.criterion(Z_a, Z_b)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)
        
        return [optimizer], [scheduler]
    
    
    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        (X, Y_a, Y_b), label = batch
        
        # get representation (without gradient computation! We don't want to train that)
        with torch.no_grad():
            if self.encoder is None:
                representation, _ = self(X, enc=1)
            else:
                representation, _ = self(X)
        
        representation = representation.detach()
        
        # use the FC layer to obtain some classification
        prediction = self.eval_fc(representation)
        loss = F.cross_entropy(prediction, label)
        
        # optimize the evaluation layer
        loss.backward()
        self.eval_optimizer.step()
        self.eval_optimizer.zero_grad()
        
        acc = accuracy(prediction, label)
        self.log("train_eval_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_eval_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    
    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        (X, Y_a, Y_b), label = batch
        
        # get representation (without gradient computation! We don't want to train that)
        with torch.no_grad():
            if self.encoder is None:
                representation, _ = self(X, enc=1)
            else:
                representation, _ = self(X)
        
            # use the FC layer to obtain some classification
            prediction = self.eval_fc(representation)
            loss = F.cross_entropy(prediction, label)
        
            acc = accuracy(prediction, label)
            self.log("val_eval_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val_eval_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        