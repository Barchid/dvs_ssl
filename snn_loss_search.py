import pytorch_lightning as pl
from project.datamodules.dvs_datamodule import DVSDataModule
from project.losses.snn_loss import SnnLoss
from project.utils.barlow_transforms import BarlowTwinsTransform
from project.ssl_module import SSLModule
import torch
import os
from matplotlib import pyplot as plt

from project.utils.eval_callback import OnlineFineTuner
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import traceback
from datetime import datetime
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 1000
learning_rate = 1e-2  # barlowsnn=0.1, vicregsnn=0.01, dvs=1e-3
timesteps = 12
batch_size = 128
dataset = "dvsgesture"
ssl_loss = "barlow_twins"
output_all = True

trans = []

datamodule = DVSDataModule(
    batch_size,
    dataset,
    timesteps,
    data_dir="data",
    barlow_transf=trans,
    in_memory=False,
    num_workers=0,
    mode="snn",
)


def objective(inv_sugg, var_sugg, cov_sugg):
    global datamodule
    try:
        module = SSLModule(
            n_classes=datamodule.num_classes,
            learning_rate=learning_rate,
            epochs=100,
            timesteps=timesteps,
            ssl_loss="snn",
            enc1="snn",
            enc2="snn",
            output_all=True,
        )

        module.criterion = SnnLoss(
            invariance_loss_weight=inv_sugg,
            variance_loss_weight=var_sugg,
            covariance_loss_weight=cov_sugg,
        )

        online_finetuner = OnlineFineTuner(
            encoder_output_dim=512,
            num_classes=datamodule.num_classes,
            output_all=True,
        )

        trainer = pl.Trainer(
            logger=True,
            checkpoint_callback=False,
            max_epochs=100,
            gpus=1 if torch.cuda.is_available() else None,
            callbacks=[
                online_finetuner,
                EarlyStopping(monitor="train_loss", mode="min"),
            ],
            precision=16,
        )
        hyperparameters = dict(
            invariance_loss_weight=inv_sugg,
            variance_loss_weight=var_sugg,
            covariance_loss_weight=cov_sugg,
        )
        trainer.logger.log_hyperparams(hyperparameters)
        trainer.fit(module, datamodule=datamodule)

        # write in score
        report = open("report_emd_prestudy.txt", "a")
        report.write(
            f"{inv_sugg} {cov_sugg} {var_sugg} {trainer.callback_metrics['online_val_acc'].item()} \n"
        )
        report.flush()
        report.close()

        return trainer.callback_metrics["online_val_acc"].item()
    except:
        traceback.print_exc()
        report = open("errors.txt", "a")
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        report.write(f"{dt_string} ===> error ! {inv_sugg} {cov_sugg} {var_sugg}\n")
        report.flush()
        report.close()

        return -1.0


if __name__ == "__main__":
    pl.seed_everything(1234)

    invs = list(np.arange(0.0, 25.1, 0.1))
    covs = list(np.arange(0.0, 25.1, 0.1))
    vars = list(np.arange(0.0, 25.1, 0.1))

    best_metr = -1.0
    best_params = [0.0, 0.0, 0.0]
    trial_num = 0
    for inv in invs:
        for cov in covs:
            for var in vars:
                metr = objective(inv, var, cov)
                if metr > best_metr:
                    best_metr = metr
                    best_params = [inv, cov, var]
                    line = f"{str(trial_num).zfill(6)} : NEW BEST {metr} ! INV={inv} COV={cov} VAR={var}\n"
                    print(line)
                    logg = open("log_prestudy.txt", "a")
                    logg.write(line)
                    logg.flush()
                    logg.close()

    print(f"BEST IS : {best_params} from trial {trial_num} with metric={best_metr}")
