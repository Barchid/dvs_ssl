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

DISP = "snn2"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 500
learning_rate = 3e-3  # barlowsnn=0.1, vicregsnn=0.01, dvs=1e-3
timesteps = 6
batch_size = 128
dataset = "dvsgesture"
output_all = False
data_dir = "/data/fox-data/datasets/spiking_camera_datasets/"


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


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
        data_dir=data_dir,
        barlow_transf=trans,
        in_memory=False,
        num_workers=0,
        mode=mode,
        use_barlow_trans=False,
    )

    module = ClassifModule(
        n_classes=datamodule.num_classes,
        learning_rate=learning_rate,
        epochs=epochs,
        timesteps=timesteps,
        mode=mode,
        output_all=output_all,
    )

    name = f"simptrain_{dataset}_{mode}"
    name += "_ALL" if output_all else ""
    for tr in trans:
        name += f"_{tr}"

    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=torch.cuda.device_count(),
        callbacks=[checkpoint_callback],
        logger=pl.loggers.TensorBoardLogger("experiments/simpletrains", name=f"{name}"),
        default_root_dir=f"experiments/simpletrains/{name}",
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
    report = open(f"report_simpletrain_{mode}.txt", "a")
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    report.write(
        f"{dt_string} {dataset} {checkpoint_callback.best_model_score} {mode} {trans}\n"
    )
    report.flush()
    report.close()
    return checkpoint_callback.best_model_score


def compare(mode):
    tran = ["background_activity", "flip_polarity"]
    main({"transforms": tran, "mode": mode, "output_all": False})

    tran = ["background_activity", "flip_polarity", "crop"]
    main({"transforms": tran, "mode": mode, "output_all": False})
    
    tran = ["flip_polarity", "crop"]
    main({"transforms": tran, "mode": mode, "output_all": False})

    tran = [
        "background_activity",
        "flip_polarity",
        "crop",
        "static_translation",
        "static_rotation",
    ]
    main({"transforms": tran, "mode": mode, "output_all": False})

    tran = [
        "background_activity",
        "flip_polarity",
        "crop",
        "dynamic_translation",
        "dynamic_rotation",
    ]
    main({"transforms": tran, "mode": mode, "output_all": False})

    tran = [
        "background_activity",
        "flip_polarity",
        "crop",
        "cutout",
    ]
    main({"transforms": tran, "mode": mode, "output_all": False})
    
    tran = [
        "background_activity",
        "flip_polarity",
        "crop",
        "cutpaste",
    ]
    main({"transforms": tran, "mode": mode, "output_all": False})
    
    tran = [
        "background_activity",
        "flip_polarity",
        "crop",
        "event_drop",
    ]
    main({"transforms": tran, "mode": mode, "output_all": False})
    
    tran = [
        "background_activity",
        "flip_polarity",
        "crop",
        "static_translation",
        "static_rotation",
        "dynamic_translation",
        "dynamic_rotation",
        "cutout",
        "cutpaste"
    ]
    main({"transforms": tran, "mode": mode, "output_all": False})


if __name__ == "__main__":
    compare(mode="snn")
    compare(mode="cnn")
    compare(mode="3dcnn")
    
    exit()
    tran = ["background_activity", "flip_polarity", "crop"]
    main({"transforms": tran, "mode": "snn", "output_all": False})

    tran = [
        "background_activity",
        "flip_polarity",
        "crop",
        "static_translation",
        "static_rotation",
    ]
    main({"transforms": tran, "mode": "snn", "output_all": False})

    tran = [
        "background_activity",
        "flip_polarity",
        "crop",
        "dynamic_translation",
        "dynamic_rotation",
    ]
    main({"transforms": tran, "mode": "snn", "output_all": False})

    tran = ["background_activity", "flip_polarity", "crop"]
    main({"transforms": tran, "mode": "snn2", "output_all": False})

    exit()
    poss_trans = list(
        powerset(["background_activity", "reverse", "flip_polarity", "crop"])
    )
    print(poss_trans)
    best_acc = -2
    best_tran = None
    for curr in poss_trans:
        acc = main({"transforms": list(curr), "mode": DISP, "output_all": False})
        if acc > best_acc:
            best_acc = acc
            best_tran = list(curr)

    messss = f"BEST BASIC FOR {DISP} IS : {best_acc} {best_tran}"
    print(messss)
    report = open(f"report_simpletrain_{DISP}.txt", "a")
    report.write(f"{messss}\n\n\n")
    report.flush()
    report.close()

    # study based on transrot
    curr = [*best_tran, "static_translation", "static_rotation"]
    st = main({"transforms": curr, "mode": DISP, "output_all": False})

    curr = [*best_tran, "dynamic_translation", "dynamic_rotation"]
    dyn = main({"transforms": curr, "mode": DISP, "output_all": False})

    if dyn >= st:
        messss = f"BEST TRANSROT FOR CNN IS DYNAMIC = {dyn}"
        best_tran = [*best_tran, "dynamic_translation", "dynamic_rotation"]
    else:
        messss = f"BEST TRANSROT FOR CNN IS STATIC = {st}"
        best_tran = [*best_tran, "static_translation", "static_rotation"]

    print(messss)
    report = open(f"report_simpletrain_{DISP}.txt", "a")
    report.write(f"{messss}\n\n\n")
    report.flush()
    report.close()

    # study on cuts
    curr = [*best_tran, "cutout"]
    cutout = main({"transforms": curr, "mode": DISP, "output_all": False})

    curr = [*best_tran, "event_drop"]
    eventdrop = main({"transforms": curr, "mode": DISP, "output_all": False})

    curr = [*best_tran, "cutpaste"]
    cutpaste = main({"transforms": curr, "mode": DISP, "output_all": False})

    curr = [*best_tran, "moving_occlusions"]
    movingocc = main({"transforms": curr, "mode": DISP, "output_all": False})

    exit()
