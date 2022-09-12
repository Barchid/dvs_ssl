import pytorch_lightning as pl
from project.datamodules.dvs_datamodule import DVSDataModule
from project.losses.snn_loss import SnnLoss
from project.utils.barlow_transforms import BarlowTwinsTransform
from project.ssl_module import SSLModule
import torch
import os
from matplotlib import pyplot as plt
from optuna.integration import PyTorchLightningPruningCallback

from project.utils.eval_callback import OnlineFineTuner
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import optuna

import traceback
from datetime import datetime

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


def objective(trial):
    global datamodule
    try:      
        module = SSLModule(
            n_classes=datamodule.num_classes,
            learning_rate=learning_rate,
            epochs=12,
            timesteps=timesteps,
            ssl_loss="snn_loss_emd",
            enc1="snn",
            enc2="snn",
            output_all=True,
        )

        inv_sugg = trial.suggest_float("inv_sugg", 0.1, 25.0)
        var_sugg = trial.suggest_float("var_sugg", 0.1, 25.0)
        cov_sugg = trial.suggest_float("cov_sugg", 0.1, 25.0)

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
            logger=False,
            checkpoint_callback=False,
            max_epochs=12,
            gpus=1 if torch.cuda.is_available() else None,
            callbacks=[
                online_finetuner,
                EarlyStopping(monitor="train_loss", mode="min", min_delta=0.1),
                # PyTorchLightningPruningCallback(trial, monitor="online_val_acc"),
            ],
            precision=16,
        )
        # hyperparameters = dict(
        #     invariance_loss_weight=inv_sugg,
        #     variance_loss_weight=var_sugg,
        #     covariance_loss_weight=cov_sugg,
        # )
        # trainer.logger.log_hyperparams(hyperparameters)
        trainer.fit(module, datamodule=datamodule)

        # write in score
        report = open("report_emd_study.txt", "a")
        report.write(
            f"ACC={trainer.callback_metrics['online_train_acc'].item()} VAL_ACC={trainer.callback_metrics['online_val_acc'].item()} INV={inv_sugg} COV={cov_sugg} VAR={var_sugg}\n"
        )
        report.flush()
        report.close()

        return trainer.callback_metrics["online_train_acc"].item()
    except:
        # traceback.print_exc()
        mess = traceback.format_exc()
        report = open('errors.txt', 'a')
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        report.write(f"{dt_string} ===> error ! {inv_sugg} {cov_sugg} {var_sugg}\n{mess}\n=========\n\n")
        report.flush()
        report.close()


if __name__ == "__main__":
    pl.seed_everything(1234)

    # pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()
    study_name = f"snn_loss_emd_v5"
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{study_name}.db",
        direction="maximize",
        # pruner=pruner,
    )
    study.optimize(objective, n_trials=1000000)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
