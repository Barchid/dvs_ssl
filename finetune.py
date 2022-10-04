import pytorch_lightning as pl
from project.datamodules.dvs_datamodule import DVSDataModule
from project.finetune_module import FinetuneModule
from project.utils.barlow_transforms import BarlowTwinsTransform
from project.ssl_module import SSLModule
import torch
import os
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from project.utils.eval_callback import OnlineFineTuner
from datetime import date

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 1000
learning_rate = 1e-2  # barlowsnn=0.1, vicregsnn=0.01, dvs=1e-3
timesteps = 12
batch_size = 128
dataset = "dvsgesture"
ssl_loss = "barlow_twins"
output_all = False


trans = ["background_activity", "reverse", "flip_polarity", "crop"]


def main(args):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",  # TODO: select the logged metric to monitor the checkpoint saving
        filename="finetuned-{epoch:03d}-{val_acc:.4f}",
        save_top_k=1,
        mode="max",
    )

    ssl_module = SSLModule.load_from_checkpoint(args["ckpt_path"])

    if ssl_module.encoder is not None:
        encoder = ssl_module.encoder
    else:
        encoder = ssl_module.encoder1

    datamodule = DVSDataModule(
        batch_size,
        dataset,
        timesteps,
        data_dir="data",
        barlow_transf=trans,
        in_memory=False,
        num_workers=0,
        mode=ssl_module.enc1,
    )

    module = FinetuneModule(
        encoder=encoder,
        n_classes=datamodule.num_classes,
        output_all=output_all,
        finetune_all=args["finetune_all"],
        mode=ssl_module.enc1,
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=torch.cuda.device_count(),
        callbacks=[checkpoint_callback],
        logger=pl.loggers.TensorBoardLogger("experiments", name=f"{date.today}"),
        default_root_dir=f"experiments/{date.today}",
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
        # traceback.print_exc()
        report = open("errors.txt", "a")
        report.write(f"{date.today} ===> error ! \n")
        report.flush()
        report.close()

    # write in score
    report = open("report.txt", "a")
    report.write(f"{dataset} {date.today} {checkpoint_callback.best_model_score} \n")
    report.flush()
    report.close()


if __name__ == "__main__":

    main()
