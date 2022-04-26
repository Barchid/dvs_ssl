from project.losses.barlow_twins_loss import BarlowTwinsLoss
from project.losses.vicreg_loss import VICRegLoss
from project.models.models import get_encoder, get_projector
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.functional import accuracy

from project.models.snn_models import get_encoder_snn

class SSLModule(pl.LightningModule):
    def __init__(self, n_classes: int, learning_rate: float, epochs: int, timesteps: int, ssl_loss: str = 'barlow_twins', network: str = 'cnn', **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['epochs', 'n_classes', 'ssl_loss', 'timesteps'])
        self.epochs = epochs
        
        if network == 'cnn':
            self.encoder = get_encoder(in_channels=2 * timesteps)
        elif network == 'snn':
            self.encoder = get_encoder_snn(in_channels=2, T=timesteps)
        else: # cnn and snn
            self.encoder_cnn = get_encoder(in_channels=2 * timesteps)
            self.encoder = get_encoder_snn(in_channels=2)
            
        self.projector = get_projector()
        
        # either barlow twins or VICReg
        if ssl_loss == 'barlow_twins':
            self.criterion = BarlowTwinsLoss()
        else:
            self.criterion = VICRegLoss()
        
        # Attributes used to evaluate the feature extractor
        self.eval_fc = nn.Linear(512, n_classes) # Linear layer used to evaluate our model (512 for ResNet-18)
        self.eval_optimizer = torch.optim.Adam(self.eval_fc.parameters())

    def forward(self, Y):
        representation = self.encoder(Y)
        Z = self.projector(representation)
        return representation, Z
    
    def shared_step(self, batch):
        (X, Y_a, Y_b), label = batch
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
            representation, _ = self(Y_a)
        
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
            representation, _ = self(X) # WARNING: we use the original input (X) for validation
        
            # use the FC layer to obtain some classification
            prediction = self.eval_fc(representation)
            loss = F.cross_entropy(prediction, label)
        
            acc = accuracy(prediction, label)
            self.log("val_eval_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val_eval_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        