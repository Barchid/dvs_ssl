import pytorch_lightning as pl
from project.ssl_module import SSLModule
from project.datamodules.ncaltech101_localization import NCALTECH101Localization
import torch
from torch.utils.data import DataLoader
import os
from matplotlib import pyplot as plt
from torch.utils.data import random_split
from argparse import ArgumentParser
from project.models.models import get_encoder, get_encoder_3d
from project.models.snn_models import get_encoder_snn
from project.utils.transform_dvs import get_frame_representation, ConcatTimeChannels
from torchvision import transforms

import traceback
from datetime import datetime
from project.localization_module import LocalizationModule

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 500
learning_rate = 3e-3  # barlowsnn=0.1, vicregsnn=0.01, dvs=1e-3
timesteps = 12
batch_size = 128
# dataset = "dvsgesture"
data_dir = "data"  # "/data/fox-data/datasets/spiking_camera_datasets/"


def main(args):
    trans = []
    ckpt = args["ckpt"]
    use_enc2 = args["use_enc2"]
    mode = args["mode"]

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_iou",  # TODO: select the logged metric to monitor the checkpoint saving
        filename="{epoch:03d}-{val_iou:.4f}",
        save_top_k=1,
        mode="max",
    )

    representation = get_frame_representation(
        None, timesteps, dataset="ncaltech101"
    )
    trans = [
        representation,
        transforms.Resize(
            (128, 128), interpolation=transforms.InterpolationMode.NEAREST
        ),
    ]
    if mode == "cnn":
        trans.append(ConcatTimeChannels())

    trans = transforms.Compose(trans)

    full_set = NCALTECH101Localization(
        transform=trans
    )
    train_len = int(len(full_set) * 0.8)
    val_len = len(full_set) - train_len # 20% of dataset
    train_set, val_set = random_split(full_set, [train_len, val_len])
    print(train_len, val_len)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=5,
    )

    if ckpt is not None:
        modu = SSLModule.load_from_checkpoint(
            ckpt,
            strict=False,
            n_classes=0,  # useless
            epochs=epochs,
            timesteps=timesteps,
        )

        if modu.encoder1 is not None:
            if use_enc2:
                backbone = modu.encoder2
                mode = modu.enc2
            else:
                backbone = modu.encoder1
                mode = modu.enc1
        else:
            backbone = modu.encoder
            mode = modu.enc1
    else:
        if mode == "snn":
            backbone = get_encoder_snn(2, timesteps, output_all=False)
        elif mode == "cnn":
            backbone = get_encoder(2 * timesteps)
        else:
            backbone = get_encoder_3d(2)
            
    module = LocalizationModule(
        learning_rate=learning_rate,
        epochs=epochs,
        encoder=backbone,
        mode=mode
    )

    name = f"localization_{mode}"
    if ckpt is not None:
        name += "_pretrained"

    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=torch.cuda.device_count(),
        callbacks=[checkpoint_callback],
        logger=pl.loggers.TensorBoardLogger("experiments/localization", name=f"{name}"),
        default_root_dir=f"experiments/localization/{name}",
        # precision=16,
    )

    try:
        trainer.fit(module, train_loader, val_loader)
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
    report = open(f"report_localization.txt", "a")
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    report.write(
        f"{dt_string} {mode} {checkpoint_callback.best_model_score} {type(ckpt)} {ckpt}\n"
    )
    report.flush()
    report.close()
    return checkpoint_callback.best_model_score


if __name__ == "__main__":
    parser = ArgumentParser("Finetune")
    parser.add_argument("--ckpt_path", default=None, type=str)
    parser.add_argument(
        "--mode", choices=["snn", "cnn", "3dcnn"], type=str, default="snn"
    )
    parser.add_argument("--use_enc2", action="store_true", default=False)
    args = parser.parse_args()

    ckpt = args.ckpt_path
    use_enc2 = args.use_enc2
    mode = args.mode

    main({"ckpt": ckpt, "use_enc2": use_enc2, "mode": mode})
