import pytorch_lightning as pl
from project.ssl_module import SSLModule
from project.datamodules.gen1_formatted import Gen1Detection
import torch
from torch.utils.data import DataLoader
import os
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from project.models.detection import SNNModule
from project.models.models import get_encoder, get_encoder_3d
from project.models.snn_models import get_encoder_snn
from project.utils.transform_dvs import get_frame_representation, ConcatTimeChannels
from torchvision import transforms
from project.models.transform_rcnn import TransformDetection

import traceback
from datetime import datetime

from pl_bolts.models.detection.faster_rcnn.faster_rcnn_module import FasterRCNN
from pl_bolts.datamodules.vocdetection_datamodule import _collate_fn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign


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
        monitor="val_acc",  # TODO: select the logged metric to monitor the checkpoint saving
        filename="{epoch:03d}-{val_acc:.4f}",
        save_top_k=1,
        mode="max",
    )

    representation = get_frame_representation(
        Gen1Detection.sensor_size, timesteps, dataset="gen1"
    )
    trans = [
        representation,
        # transforms.Resize(
        #     (128, 128), interpolation=transforms.InterpolationMode.NEAREST
        # ),
    ]
    if mode == "cnn":
        trans.append(ConcatTimeChannels())

    trans = transforms.Compose(trans)

    train_set = Gen1Detection(save_to="/datas/sandbox", subset="train", transform=trans)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=_collate_fn,
    )
    val_set = Gen1Detection(save_to="/datas/sandbox", subset="test", transform=trans)
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=_collate_fn,
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

    encoder = SNNModule(backbone, mode=mode)
    anchor_generator = AnchorGenerator(sizes=((8, 16, 32, 64, 128),), aspect_ratios=((0.5, 1.0, 2.0),))
    roialign = MultiScaleRoIAlign("[0]", output_size=4, sampling_ratio=2)
    module = FasterRCNN(
        num_classes=2,
        backbone=encoder,
        image_mean=(0.0, 0.0, 0.0),
        image_std=(1.0, 1.0, 1.0),
        max_size=128,
        min_size=128,
        rpn_anchor_generator=anchor_generator, 
        box_roi_pool=roialign,
        fpn=False
    )
    module.model.transform = TransformDetection(
        128, 128, timesteps=None if mode == "cnn" else timesteps
    )

    name = f"detection_{mode}"
    if ckpt is not None:
        name += "_pretrained"

    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=torch.cuda.device_count(),
        callbacks=[checkpoint_callback],
        logger=pl.loggers.TensorBoardLogger("experiments/detection", name=f"{name}"),
        default_root_dir=f"experiments/detection/{name}",
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
    report = open(f"report_detection.txt", "a")
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
