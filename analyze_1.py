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
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from project.utils.eval_callback import OnlineFineTuner
import json
import traceback
from datetime import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 1000
learning_rate = 3e-3  # barlowsnn=0.1, vicregsnn=0.01, dvs=1e-3
timesteps = 12
batch_size = 128
# dataset = "dvsgesture"
data_dir = "/sandbox0/sami/data" #"/data/fox-data/datasets/spiking_camera_datasets/"  # "data"


def main(args):
    trans = []
    ckpt = args["ckpt"]
    src_dataset = args["src_dataset"]
    dest_dataset = args["dest_dataset"]
    use_enc2 = args["use_enc2"]
    target_dir = args["target_dir"]

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
        module = ClassifModule.load_from_checkpoint(
            ckpt,
            strict=False,
            n_classes=dest_num_classes,
            epochs=epochs,
            timesteps=timesteps,
        ).to(device)

    datamodule = DVSDataModule(
        1,
        dest_dataset,
        timesteps,
        data_dir=data_dir,
        barlow_transf=trans,
        in_memory=False,
        num_workers=0,
        mode=module.mode,
        use_barlow_trans=True,
        subset_len=None,
    )

    try:
        datamodule.setup()
        val_loader = datamodule.val_dataloader()

        with torch.no_grad():
            embeddings_matrix = torch.zeros((len(val_loader.dataset), 512))
            stem_matrix = torch.zeros((len(val_loader.dataset), 262144))
            res2_matrix = torch.zeros((len(val_loader.dataset), 65536))
            res3_matrix = torch.zeros((len(val_loader.dataset), 32768))
            res4_matrix = torch.zeros((len(val_loader.dataset), 16384))
            predictions = torch.zeros(len(val_loader.dataset))
            gts = torch.zeros(len(val_loader.dataset))
            
            module.eval()
            for idx, (inputs, label) in enumerate(val_loader):
                print('Batch n°', idx)

                (X, Y_a, Y_b) = inputs
                X = X.to(device)
                (
                    y_hat,
                    feats,
                    stem_feat,
                    res2_feat,
                    res3_feat,
                    res4_feat,
                ) = module.forward_analyze(X)
                y_hat = y_hat.squeeze()
                pred = torch.argmax(y_hat).cpu()

                feats = feats.squeeze().cpu()
                stem_feat = stem_feat.squeeze().cpu()
                res2_feat = res2_feat.squeeze().cpu()
                res3_feat = res3_feat.squeeze().cpu()
                res4_feat = res4_feat.squeeze().cpu()

                embeddings_matrix[idx] = feats
                stem_matrix[idx] = stem_feat
                res2_matrix[idx] = res2_feat
                res3_matrix[idx] = res3_feat
                res4_matrix[idx] = res4_feat
                gts[idx] = label[0]
                predictions[idx] = pred


            tosave = {
                "stem": stem_matrix,
                "res2": res2_matrix,
                "res3": res3_matrix,
                "res4": res4_matrix,
                "embeddings": embeddings_matrix,
                "predictions": predictions,
                "gts": gts
            }
            
            torch.save(tosave, f"{target_dir}/cka_stats.pt")

    except:
        # traceback.print_exc()
        mess = traceback.format_exc()
        report = open("errors.txt", "a")
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        report.write(f"{dt_string} ===> {mess}\n=========\n\n")
        report.flush()
        report.close()


if __name__ == "__main__":
    parser = ArgumentParser("Finetune")
    parser.add_argument("ckpt_path", default=None, type=str)
    parser.add_argument("--src_dataset", required=True, type=str)
    parser.add_argument("--dest_dataset", default=None, type=str)
    parser.add_argument("--subset_len", default=None, type=str, choices=["10", "25"])
    parser.add_argument("--use_enc2", action="store_true", default=False)
    parser.add_argument('--target_dir', default="experiments/results", type=str)
    args = parser.parse_args()

    ckpt = args.ckpt_path
    src_dataset = args.src_dataset
    dest_dataset = args.dest_dataset
    if dest_dataset is None:
        dest_dataset = src_dataset

    use_enc2 = args.use_enc2

    main(
        {
            "ckpt": ckpt,
            "src_dataset": src_dataset,
            "dest_dataset": dest_dataset,
            "use_enc2": use_enc2,
            "target_dir": args.target_dir
        }
    )

    # compare(mode="cnn", ckpt=ckpt, src_dataset=src_dataset, dest_dataset=dest_dataset, subs)
    # compare(mode="snn")
    # compare(mode="3dcnn")
