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

# import traceback

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 1000
learning_rate = 1e-2  # barlowsnn=0.1, vicregsnn=0.01, dvs=1e-3
timesteps = 12
batch_size = 128
dataset = "dvsgesture"
ssl_loss = "barlow_twins"
output_all = True

trans = []


def objective(trial):
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

    inv_sugg = trial.suggest_float("inv_sugg", 0.5, 25.0)
    var_sugg = trial.suggest_float("var_sugg", 0.5, 25.0)
    cov_sugg = trial.suggest_float("cov_sugg", 0.5, 25.0)

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
            EarlyStopping(monitor="val_loss", mode="min"),
            PyTorchLightningPruningCallback(trial, monitor="online_val_acc"),
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

    return trainer.callback_metrics["online_val_acc"].item()


if __name__ == "__main__":
    pl.seed_everything(1234)

    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()
    study_name = f"snn_loss_emd_v1"
    study = optuna.create_study(study_name=study_name, storage=f"sqlite:///{study_name}.db", direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=10000)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
