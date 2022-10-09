from project.losses.barlow_twins_loss import BarlowTwinsLoss
from project.losses.snn_loss import SnnLoss
from project.losses.vicreg_loss import VICRegLoss
from project.models.models import get_encoder, get_encoder_3d, get_projector
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from einops import rearrange
from torchmetrics.functional import accuracy
from spikingjelly.clock_driven import functional

from project.models.snn_models import ProjectorSSL, get_encoder_snn, get_projector_lif, try_projector
from project.models.utils import MeanSpike

class SSLModule(pl.LightningModule):
    def __init__(self, n_classes: int, learning_rate: float, epochs: int, timesteps: int, ssl_loss: str = 'barlow_twins', enc1: str = 'cnn', enc2: str = 'cnn', output_all: bool = False, multiple_proj: bool = False, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['epochs', 'n_classes', 'ssl_loss', 'timesteps'])
        self.epochs = epochs
        self.enc1 = enc1
        self.enc2 = enc2
        self.output_all = output_all
        
        self.encoder = None
        self.projector = None
        self.encoder1 = None
        self.encoder2 = None
        
        # this is ugly af but nvm
        if output_all:
            if multiple_proj:
                self.projector = ProjectorSSL()
            else:
                # self.projector = get_projector_lif()
                self.projector = try_projector()
        else:
            self.projector = get_projector()
        
        if enc1 == enc2:
            if enc1 == 'cnn':
                self.encoder = get_encoder(in_channels=2 * timesteps)
            elif enc1 == 'snn':
                self.encoder = get_encoder_snn(2, timesteps, output_all=output_all)
            elif enc1 == "3dcnn":
                self.encoder = get_encoder_3d(in_channels=2)
        else:
            if enc1 == 'cnn':
                self.encoder1 = get_encoder(in_channels=2 * timesteps)
            elif enc1 == 'snn':
                self.encoder1 = get_encoder_snn(2, timesteps, output_all=output_all)
            elif enc1 == "3dcnn":
                self.encoder1 = get_encoder_3d(in_channels=2)
                
            if enc2 == 'cnn':
                self.encoder2 = get_encoder(in_channels=2 * timesteps)
            elif enc2 == 'snn':
                self.encoder2 = get_encoder_snn(2, timesteps, output_all=output_all)
            elif enc2 == "3dcnn":
                self.encoder2 = get_encoder_3d(in_channels=2)
            
        # either barlow twins or VICReg
        if ssl_loss == 'barlow_twins':
            self.criterion = BarlowTwinsLoss()
        elif ssl_loss == "snn_loss_emd":
            self.criterion = SnnLoss(multiple_proj=multiple_proj)
        elif ssl_loss == "snn_loss_mse":
            self.criterion = SnnLoss(invariance_mode="mse", multiple_proj=multiple_proj)
        else:
            self.criterion = VICRegLoss()

    def forward(self, Y, enc=None, mode="cnn"):
        if mode == "snn":                     # 0,1,2,3,4
            Y = Y.permute(1, 0, 2, 3, 4)# from (B,T,C,H,W) to (T, B, C, H, W)
            
            if enc is None:
                functional.reset_net(self.encoder)
            elif enc == 1:
                functional.reset_net(self.encoder1)
            else:
                functional.reset_net(self.encoder2)
                
            # if self.output_all:
            #     functional.reset_net(self.projector)
            
        if mode == "cnn" and len(Y.shape) == 5:
            Y = rearrange(
                Y, "batch time channel height width -> batch (time channel) height width"
            )
            
        if mode == "3dcnn":
            Y = Y.permute(0, 2, 1, 3, 4) # from (B,T,C,H,W) to (B,C,T,H,W)
            
        if enc is None:    
            representation = self.encoder(Y)
            
        elif enc == 1:
            representation = self.encoder1(Y)
        else:
            representation = self.encoder2(Y)
            
        return representation
    
    def shared_step(self, batch):
        (X, Y_a, Y_b), label = batch
        
        if self.encoder is None:
            representation = self(Y_a, enc=1, mode=self.enc1)
            # print(torch.unique(representation))
            Z_a = self.projector(representation)
            representation = self(Y_b, enc=2, mode=self.enc2)
            Z_b = self.projector(representation)
        else:
            representation = self(Y_a, mode=self.enc1)
            Z_a = self.projector(representation)
            representation = self(Y_b, mode=self.enc1)
            Z_b = self.projector(representation)
            
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
    
    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True) # better perf
    
    
    # def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
    #     (X, Y_a, Y_b), label = batch
        
    #     # get representation (without gradient computation! We don't want to train that)
    #     with torch.no_grad():
    #         if self.encoder is None:
    #             representation, _ = self(X, enc=1, mode=self.enc1)
    #         else:
    #             representation, _ = self(X, mode=self.enc1)
        
    #     representation = representation.detach()
        
    #     # use the FC layer to obtain some classification
    #     prediction = self.eval_fc(representation)
    #     loss = F.cross_entropy(prediction, label)
        
    #     # optimize the evaluation layer
    #     loss.backward()
    #     self.eval_optimizer.step()
    #     self.eval_optimizer.zero_grad()
        
    #     acc = accuracy(prediction, label)
    #     self.log("train_eval_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
    #     self.log("train_eval_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    
    # def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
    #     (X, Y_a, Y_b), label = batch
        
    #     # get representation (without gradient computation! We don't want to train that)
    #     with torch.no_grad():
    #         if self.encoder is None:
    #             representation, _ = self(X, enc=1, mode=self.enc1)
    #         else:
    #             representation, _ = self(X, mode=self.enc1)
        
    #         # use the FC layer to obtain some classification
    #         prediction = self.eval_fc(representation)
    #         loss = F.cross_entropy(prediction, label)
        
    #         acc = accuracy(prediction, label)
    #         self.log("val_eval_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
    #         self.log("val_eval_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        