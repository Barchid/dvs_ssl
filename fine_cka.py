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
epochs = 1000
learning_rate = 3e-3  # barlowsnn=0.1, vicregsnn=0.01, dvs=1e-3
timesteps = 12
batch_size = 128
# dataset = "dvsgesture"
data_dir = "/data/fox-data/datasets/spiking_camera_datasets/"  # "data"


def main(args):
    trans = []
    subset_len = args["subset_len"]
    ckpt = args["ckpt"]
    src_dataset = args["src_dataset"]
    dest_dataset = args["dest_dataset"]
    use_enc2 = args["use_enc2"]

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",  # TODO: select the logged metric to monitor the checkpoint saving
        filename="{epoch:03d}-{val_acc:.4f}",
        save_top_k=1,
        mode="max",
    )

    src_num_classes = 10
    if src_dataset == "daily_action_dvs":
        src_num_classes = 12
    elif src_dataset == "n-caltech101":
        src_num_classes = 101
    elif src_dataset == "ncars":
        src_num_classes = 2
    elif src_dataset == "dvsgesture":
        src_num_classes = 11
    # elif src_dataset == "ncars":
    #     src_num_classes = 2

    dest_num_classes = 10
    if dest_dataset == "daily_action_dvs":
        dest_num_classes = 12
    elif dest_dataset == "n-caltech101":
        dest_num_classes = 101
    elif dest_dataset == "ncars":
        dest_num_classes = 2
    elif dest_dataset == "dvsgesture":
        dest_num_classes = 11
    # elif dest_dataset == "ncars":
    #     dest_num_classes = 2

    if ckpt is not None:
        modu = SSLModule.load_from_checkpoint(
            ckpt,
            strict=False,
            n_classes=src_num_classes,
            epochs=epochs,
            timesteps=timesteps,
        )

    module = ClassifModule(
        n_classes=dest_num_classes,
        learning_rate=learning_rate,
        epochs=epochs,
        timesteps=timesteps,
        mode=modu.enc1,
    )

    datamodule = DVSDataModule(
        batch_size,
        dest_dataset,
        timesteps,
        data_dir=data_dir,
        barlow_transf=trans,
        in_memory=False,
        num_workers=0,
        mode=modu.enc1,
        use_barlow_trans=True,
        subset_len=subset_len,
    )

    if ckpt is not None:
        modu = SSLModule.load_from_checkpoint(
            ckpt,
            strict=False,
            n_classes=src_num_classes,
            epochs=epochs,
            timesteps=timesteps,
        )

        if modu.encoder1 is not None:
            if use_enc2:
                enco = modu.encoder1
            else:
                enco = modu.encoder2
        else:
            enco = modu.encoder

        module.encoder = enco

        module.encoder.requires_grad_(False)

    name = f"{src_dataset}_{dest_dataset}_{modu.enc1}_{modu.enc2}"
    for tr in trans:
        name += f"_{tr}"

    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=torch.cuda.device_count(),
        callbacks=[checkpoint_callback],
        logger=pl.loggers.TensorBoardLogger("experiments/ckas", name=f"{name}"),
        default_root_dir=f"experiments/ckas/{name}",
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
    report = open(f"report_cka_{modu.enc1}_{modu.enc2}.txt", "a")
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    report.write(
        f"{dt_string} {src_dataset} {dest_dataset} {subset_len} {checkpoint_callback.best_model_score} {modu.enc1} {modu.enc2} {trans} {type(ckpt)}\n"
    )
    report.flush()
    report.close()

    return checkpoint_callback.best_model_score


def compare(mode, ckpt=None):
    # main({"mode": mode, "subset_len": "10%", "ckpt": None})

    # main({"mode": mode, "subset_len": "25%", "ckpt": None})

    main({"mode": mode, "subset_len": "10%", "ckpt": ckpt})

    main({"mode": mode, "subset_len": "25%", "ckpt": ckpt})


if __name__ == "__main__":
    parser = ArgumentParser("Finetune")
    parser.add_argument("ckpt_path", default=None, type=str)
    parser.add_argument("--src_dataset", required=True, type=str)
    parser.add_argument("--dest_dataset", default=None, type=str)
    parser.add_argument("--subset_len", default=None, type=str, choices=["10", "25"])
    parser.add_argument("--use_enc2", action="store_true", default=False)
    args = parser.parse_args()

    ckpt = args.ckpt_path
    src_dataset = args.src_dataset
    dest_dataset = args.dest_dataset
    if dest_dataset is None:
        dest_dataset = src_dataset
    subset_len = args.subset_len
    if subset_len is not None:
        subset_len = subset_len + "%"

    use_enc2 = args.use_enc2

    main(
        {
            "subset_len": subset_len,
            "ckpt": ckpt,
            "src_dataset": src_dataset,
            "dest_dataset": dest_dataset,
            "use_enc2": use_enc2,
        }
    )

    # compare(mode="cnn", ckpt=ckpt, src_dataset=src_dataset, dest_dataset=dest_dataset, subs)
    # compare(mode="snn")
    # compare(mode="3dcnn")
