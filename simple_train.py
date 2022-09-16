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
epochs = 100
learning_rate = 1e-2  # barlowsnn=0.1, vicregsnn=0.01, dvs=1e-3
timesteps = 12
batch_size = 128
dataset = "dvsgesture"
output_all = False

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def main(args):
    trans = args["transforms"]
    mode = args["mode"]
    output_all = args["output_all"]

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
        data_dir="data",
        barlow_transf=trans,
        in_memory=False,
        num_workers=0,
        mode=mode,
    )

    module = ClassifModule(
        n_classes=datamodule.num_classes,
        learning_rate=learning_rate,
        epochs=epochs,
        timesteps=timesteps,
        mode=mode,
        output_all=output_all,
    )

    name = f"{dataset}_{mode}"
    name += "_ALL" if output_all else ""
    for tr in trans:
        name += f"_{tr}"

    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=torch.cuda.device_count(),
        callbacks=[checkpoint_callback],
        logger=pl.loggers.TensorBoardLogger(
            "experiments", name=f"{name}"
        ),
        default_root_dir=f"experiments/{name}",
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
    report = open("report_simpletrain.txt", "a")
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    report.write(
        f"{dt_string} {dataset} {checkpoint_callback.best_model_score} {mode} {output_all} {trans}\n"
    )
    report.flush()
    report.close()
    return checkpoint_callback.best_model_score


if __name__ == "__main__":
    pl.seed_everything(1234)

    poss_trans = list(powerset(["flip", "background_activity", "reverse", "flip_polarity", "crop"]))
    print(poss_trans)
    best_acc = -2
    best_tran = None
    for curr in poss_trans:
        acc = main({"transforms": list(curr), "mode": "cnn", "output_all": False})
        if acc > best_acc:
            best_acc = acc
            best_tran = list(curr)
                
    messss = f'BEST BASIC FOR CNN IS : {best_acc} {best_tran}'
    print(messss)
    report = open("report_simpletrain.txt", "a")
    report.write(f"{messss}\n\n\n")
    report.flush()
    report.close()
    
    # study based on transrot
    curr = [*best_tran, "static_translation", "static_rotation"]
    st = main({"transforms": curr, "mode": "cnn", "output_all": False})
    
    curr = [*best_tran, "dynamic_translation", "dynamic_rotation"]
    dyn = main({"transforms": curr, "mode": "cnn", "output_all": False})
    
    if dyn >= st:
        messss = f"BEST TRANSROT FOR CNN IS DYNAMIC = {dyn}"
        best_tran = [*best_tran, "dynamic_translation", "dynamic_rotation"]
    else:
        messss = f"BEST TRANSROT FOR CNN IS STATIC = {st}"
        best_tran = [*best_tran, "static_translation", "static_rotation"]
    
    print(messss)
    report = open("report_simpletrain.txt", "a")
    report.write(f"{messss}\n\n\n")
    report.flush()
    report.close()
    
    # study on cuts
    curr = [*best_tran, "cutout"]
    cutout = main({"transforms": curr, "mode": "cnn", "output_all": False})
    
    curr = [*best_tran, "event_drop"]
    eventdrop = main({"transforms": curr, "mode": "cnn", "output_all": False})
    
    curr = [*best_tran, "cutpaste"]
    cutpaste = main({"transforms": curr, "mode": "cnn", "output_all": False})
    
    curr = [*best_tran, "moving_occlusions"]
    movingocc = main({"transforms": curr, "mode": "cnn", "output_all": False})
    
    