from itertools import chain, combinations
from operator import mod
import pytorch_lightning as pl
from project.classif_module import ClassifModule
from project.datamodules.dvs_datamodule import DVSDataModule
from project.finetune_module import FinetuneModule
from project.utils.barlow_transforms import BarlowTwinsTransform
from project.ssl_module import SSLModule
import torch
import os
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from project.utils.eval_callback import OnlineFineTuner

import traceback
from datetime import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 500
learning_rate = 3e-3  # barlowsnn=0.1, vicregsnn=0.01, dvs=1e-3
timesteps = 12
batch_size = 128
dataset = "dvsgesture"
data_dir = "data"  # "/data/fox-data/datasets/spiking_camera_datasets/"


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def main(args):
    trans = []
    mode = args["mode"]
    subset_len = args["subset_len"]
    ckpt = args["ckpt"]

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",  # TODO: select the logged metric to monitor the checkpoint saving
        filename=mode + "-{epoch:03d}-{val_acc:.4f}",
        save_top_k=1,
        mode="max",
    )

    datamodule = DVSDataModule(
        batch_size,
        dataset,
        timesteps,
        data_dir=data_dir,
        barlow_transf=trans,
        in_memory=False,
        num_workers=0,
        mode=mode,
        use_barlow_trans=True,
        subset_len=subset_len,
    )


    module = ClassifModule(
        n_classes=datamodule.num_classes,
        learning_rate=learning_rate,
        epochs=epochs,
        timesteps=timesteps,
        mode=mode,
    )
    
    if ckpt is not None:
        chek = torch.load(ckpt)
        state_dict = chek['state_dict']
        print(state_dict)
        exit()
        
        modu = SSLModule.load_from_checkpoint(
            ckpt, strict=False, n_classes=datamodule.num_classes, epochs=epochs, timesteps=timesteps
        )
        
        if modu.encoder1 is not None:
            enco = modu.encoder1
        else:
            enco = modu.encoder
            
        module.encoder = enco

    name = f"semisup_{dataset}_{mode}"
    for tr in trans:
        name += f"_{tr}"

    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=torch.cuda.device_count(),
        callbacks=[checkpoint_callback],
        logger=pl.loggers.TensorBoardLogger("experiments/semisups", name=f"{name}"),
        default_root_dir=f"experiments/semisups/{name}",
        precision=16,
    )

    try:
        trainer.fit(module, datamodule=datamodule)
    except:
        # traceback.print_exc()
        mess = traceback.format_exc()
        report = open("errors.txt", "a")
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        report.write(f"{dt_string} ===> {mess}\n=========\n\n")
        report.flush()
        report.close()
        return -1

    # write in score
    report = open(f"report_semisupervised_{mode}.txt", "a")
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    report.write(
        f"{dt_string} {dataset} {subset_len} {checkpoint_callback.best_model_score} {mode} {trans} {type(ckpt)}\n"
    )
    report.flush()
    report.close()
    return checkpoint_callback.best_model_score


def compare(mode, ckpt=None):
    main({"mode": mode, "subset_len": "10%", "ckpt": None})

    main({"mode": mode, "subset_len": "25%", "ckpt": None})
    
    main({"mode": mode, "subset_len": "10%", "ckpt": ckpt})
    
    main({"mode": mode, "subset_len": "25%", "ckpt": ckpt})


if __name__ == "__main__":
    parser = ArgumentParser("Finetune")
    parser.add_argument("ckpt_path", default=None, type=str)
    args = parser.parse_args()

    ckpt = args.ckpt_path

    compare(mode="cnn", ckpt=ckpt)
    # compare(mode="snn")
    # compare(mode="3dcnn")
