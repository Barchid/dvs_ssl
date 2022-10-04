import pytorch_lightning as pl
from project.datamodules.dvs_datamodule import DVSDataModule
from project.utils.barlow_transforms import BarlowTwinsTransform
from project.ssl_module import SSLModule
import torch
import numpy as np
from itertools import chain, combinations
import os
from matplotlib import pyplot as plt

import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from project.utils.eval_callback import OnlineFineTuner
import traceback
from datetime import datetime
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 1000
learning_rate = 1e-2  # barlowsnn=0.1, vicregsnn=0.01, dvs=1e-3
timesteps = 12
batch_size = 128
dataset = "dvsgesture"
ssl_loss = "barlow_twins"
output_all = False

trans = ["background_activity", "reverse", "flip_polarity", "crop"]


def create_features(args):
    filename = args["filename"]
    ssl_module = SSLModule.load_from_checkpoint(args["ckpt_path"])
    ssl_module.eval()
    datamodule = DVSDataModule(
        1,
        dataset,
        timesteps,
        data_dir="data",
        barlow_transf=trans,
        in_memory=False,
        num_workers=0,
        mode=ssl_module.enc1,
    )
    datamodule.setup()

    all_features = []
    all_labels = []
    with torch.no_grad():
        i = 0
        for batch in datamodule.val_dataloader():
            (X, Y_a, Y_b), label = batch
            features = ssl_module(X)  # shape=(1,512)
            features = features.squeeze().numpy()  # (512)
            all_features.append(features)
            all_labels.append(label[0].item())
            print(f"Processing sample n°{str(i).zfill(4)}")
            i += 1

    # create npy files
    all_features = np.asarray(all_features)
    all_labels = np.asarray(all_labels)
    np.save(f"{filename}_features.npy", all_features)
    np.save(f"{filename}_labels.npy", all_labels)


def tsne_visualize(args):
    filename = args["filename"]
    features_file = f"{filename}_features.npy"
    labels_file = f"{filename}_labels.npy"

    features = np.load(features_file)
    labels = np.load(labels_file)

    # T-SNE
    pca = PCA(n_components=50)
    X = pca.fit_transform(features)
    # Parameters to tune
    tsne = TSNE(n_components=2)  # , perplexity=50, n_iter=10000, learning_rate=1000)
    tsne_result = tsne.fit_transform(X)
    tsne_result_df = pd.DataFrame(
        {"x": tsne_result[:, 0], "y": tsne_result[:, 1], "label": labels}
    )
    fig, ax = plt.subplots(1)
    sns.scatterplot(x="x", y="y", hue="label", data=tsne_result_df, ax=ax, s=120)
    lim = (tsne_result.min() - 5, tsne_result.max() + 5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    fig.savefig(f"plots/{filename}.png")


if __name__ == "__main__":
    pl.seed_everything(1234)

    create_features({"ckpt_path": "coucou"})
