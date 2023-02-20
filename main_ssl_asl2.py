import pytorch_lightning as pl
from project.datamodules.dvs_datamodule import DVSDataModule
from project.utils.barlow_transforms import BarlowTwinsTransform
from project.ssl_module import SSLModule
import torch
from itertools import chain, combinations
import os
from matplotlib import pyplot as plt

from project.utils.eval_callback import OnlineFineTuner
import traceback
from datetime import datetime
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 500
learning_rate = 1e-2  # barlowsnn=0.1, vicregsnn=0.01, dvs=1e-3
timesteps = 12
batch_size = 128
dataset = "asl-dvs"


def main(args):
    trans = args["transforms"]
    mode = args["mode"]
    output_all = args["output_all"]
    ssl_loss = args["ssl_loss"]

    name = f"{dataset}_{mode}_DIFFE"
    name += "_ALL" if output_all else ""
    for tr in trans:
        name += f"_{tr}"

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="online_val_acc_enc1",  # TODO: select the logged metric to monitor the checkpoint saving
        filename=name + "-{epoch:03d}-{online_val_acc_enc1:.4f}",
        save_top_k=1,
        mode="max",
    )
    
    checkpoint_callback2 = pl.callbacks.ModelCheckpoint(
        monitor="online_val_acc_enc2",  # TODO: select the logged metric to monitor the checkpoint saving
        filename=name + "-ENC2-{epoch:03d}-{online_val_acc_enc2:.4f}",
        save_top_k=1,
        mode="max",
    )

    datamodule = DVSDataModule(
        batch_size,
        dataset,
        timesteps,
        data_dir="/sandbox0/sami/data",
        barlow_transf=trans,
        in_memory=False,
        num_workers=0,
        mode="snn" if mode=="cnn" else mode,
    )

    lr = learning_rate

    module = SSLModule(
        n_classes=datamodule.num_classes,
        learning_rate=lr,
        epochs=epochs,
        ssl_loss=ssl_loss,
        timesteps=timesteps,
        enc1="snn",
        enc2=mode,
        output_all=output_all,
        multiple_proj=False,
    )

    online_finetuner = OnlineFineTuner(
        encoder_output_dim=512,
        num_classes=datamodule.num_classes,
        output_all=output_all,
    )
    
    online_finetuner2 = OnlineFineTuner(
        encoder_output_dim=512,
        num_classes=datamodule.num_classes,
        output_all=output_all,
        enc="enc2"
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=torch.cuda.device_count(),
        callbacks=[
            online_finetuner,
            online_finetuner2,
            checkpoint_callback,
            checkpoint_callback2,
            # EarlyStopping(monitor="online_val_acc", mode="max", patience=75),
        ],
        # logger=pl.loggers.TensorBoardLogger("experiments", name=name),
        default_root_dir=f"/sandbox0/sami/experiments/{name}",
        precision=16,
    )

    # lr_finder = trainer.tuner.lr_find(module, datamodule=datamodule)
    # fig = lr_finder.plot(suggest=True)
    # fig.savefig('lr.png')   # save the figure to file
    # plt.close(fig)    # close th
    # print(f'SUGGESTION IS :', lr_finder.suggestion())
    # exit()
    try:
        trainer.fit(module, datamodule=datamodule)
    except:
        mess = traceback.format_exc()
        report = open("errors.txt", "a")
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        report.write(f"{dt_string} ===> {mess}\n=========\n\n")
        report.flush()
        report.close()
        return -1

    # write in score
    report = open(f"report_snn_{mode}.txt", "a")
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    report.write(
        f"{dt_string} {dataset} ENC1={checkpoint_callback.best_model_score} ENC2={checkpoint_callback2.best_model_score} {mode} {output_all} {trans}\n"
    )
    report.flush()
    report.close()
    return checkpoint_callback.best_model_score, checkpoint_callback2.best_model_score


def compare(mode):
    # tran = ["background_activity", "flip_polarity", "crop"]
    # acc = main(
    #     {"transforms": tran, "ssl_loss": "vicreg", "mode": mode, "output_all": False}
    # )
    
    # tran = ["background_activity", "flip_polarity", "crop", "reverse"]
    # acc = main(
    #     {"transforms": tran, "ssl_loss": "vicreg", "mode": mode, "output_all": False}
    # )
    
    
    tran = ["background_activity", "flip_polarity", "crop", "reverse", "event_drop_2"]
    acc = main(
        {"transforms": tran, "ssl_loss": "vicreg", "mode": mode, "output_all": False}
    )


if __name__ == "__main__":
    compare(mode="cnn")
    compare(mode="3dcnn")