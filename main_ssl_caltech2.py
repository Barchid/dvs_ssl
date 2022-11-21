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
dataset = "n-caltech101"


def main(args):
    trans = args["transforms"]
    mode = args["mode"]
    output_all = args["output_all"]
    ssl_loss = args["ssl_loss"]

    name = f"{dataset}_{mode}"
    name += "_ALL" if output_all else ""
    for tr in trans:
        name += f"_{tr}"

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="online_val_acc",  # TODO: select the logged metric to monitor the checkpoint saving
        filename=name + "-{epoch:03d}-{online_val_acc:.4f}",
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

    lr = learning_rate

    module = SSLModule(
        n_classes=datamodule.num_classes,
        learning_rate=lr,
        epochs=epochs,
        ssl_loss=ssl_loss,
        timesteps=timesteps,
        enc1=mode,
        enc2=mode,
        output_all=output_all,
        multiple_proj=False,
    )

    online_finetuner = OnlineFineTuner(
        encoder_output_dim=512,
        num_classes=datamodule.num_classes,
        output_all=output_all,
        enc=None,
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=torch.cuda.device_count(),
        callbacks=[
            online_finetuner,
            checkpoint_callback,
            # EarlyStopping(monitor="online_val_acc", mode="max", patience=75),
        ],
        # logger=pl.loggers.TensorBoardLogger("experiments", name=name),
        default_root_dir=f"experiments/{name}",
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
    report = open(f"report_{mode}.txt", "a")
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    report.write(
        f"{dt_string} {dataset} {checkpoint_callback.best_model_score} {mode} {output_all} {trans}\n"
    )
    report.flush()
    report.close()
    return checkpoint_callback.best_model_score


def compare(mode):
    tran = ["background_activity", "flip_polarity", "crop", "reverse", "event_drop_2"]
    acc = main(
        {"transforms": tran, "ssl_loss": "vicreg", "mode": mode, "output_all": False}
    )


if __name__ == "__main__":
    compare(mode="cnn")

    exit()

    tran = ["background_activity", "flip_polarity", "crop", "transrot"]
    acc = main(
        {"transforms": tran, "ssl_loss": "vicreg", "mode": "cnn", "output_all": False}
    )

    tran = ["background_activity", "flip_polarity", "crop", "transrot", "cutpaste"]
    acc = main(
        {"transforms": tran, "ssl_loss": "vicreg", "mode": "cnn", "output_all": False}
    )

    tran = ["background_activity", "flip_polarity", "crop", "transrot", "cutout"]
    acc = main(
        {"transforms": tran, "ssl_loss": "vicreg", "mode": "cnn", "output_all": False}
    )

    #################
    tran = [
        "background_activity",
        "flip_polarity",
        "crop",
        "dynamic_translation",
        "dynamic_rotation",
    ]
    acc = main(
        {"transforms": tran, "ssl_loss": "vicreg", "mode": "3dcnn", "output_all": False}
    )

    tran = ["background_activity", "flip_polarity", "crop", "transrot", "event_drop"]
    acc = main(
        {"transforms": tran, "ssl_loss": "vicreg", "mode": "3dcnn", "output_all": False}
    )

    tran = ["background_activity", "flip_polarity", "crop", "transrot", "event_drop_2"]
    acc = main(
        {"transforms": tran, "ssl_loss": "vicreg", "mode": "3dcnn", "output_all": False}
    )

    tran = ["background_activity", "flip_polarity", "crop", "transrot", "cutout"]
    acc = main(
        {"transforms": tran, "ssl_loss": "vicreg", "mode": "3dcnn", "output_all": False}
    )

    tran = ["background_activity", "flip_polarity", "crop", "transrot", "cutpaste"]
    acc = main(
        {"transforms": tran, "ssl_loss": "vicreg", "mode": "3dcnn", "output_all": False}
    )

    ################
    tran = [
        "background_activity",
        "reverse",
        "flip_polarity",
        "crop",
        "dynamic_translation",
        "dynamic_rotation",
    ]
    acc = main(
        {"transforms": tran, "ssl_loss": "vicreg", "mode": "snn", "output_all": False}
    )
    tran = [
        "background_activity",
        "reverse",
        "flip_polarity",
        "crop",
        "transrot",
        "cutout",
    ]
    acc = main(
        {"transforms": tran, "ssl_loss": "vicreg", "mode": "snn", "output_all": False}
    )
    tran = [
        "background_activity",
        "reverse",
        "flip_polarity",
        "crop",
        "transrot",
        "cutpaste",
    ]
    acc = main(
        {"transforms": tran, "ssl_loss": "vicreg", "mode": "snn", "output_all": False}
    )
    tran = [
        "background_activity",
        "reverse",
        "flip_polarity",
        "crop",
        "transrot",
        "event_drop_2",
    ]
    acc = main(
        {"transforms": tran, "ssl_loss": "vicreg", "mode": "snn", "output_all": False}
    )

    ##########

    exit()

    tran = ["background_activity", "reverse", "flip_polarity", "crop", "transrot"]
    acc = main(
        {"transforms": tran, "ssl_loss": "vicreg", "mode": "snn", "output_all": False}
    )

    tran = [
        "background_activity",
        "reverse",
        "flip_polarity",
        "crop",
        "dynamic_translation",
        "dynamic_rotation",
        "event_drop_2",
    ]
    acc = main(
        {"transforms": tran, "ssl_loss": "vicreg", "mode": "snn", "output_all": False}
    )

    tran = [
        "background_activity",
        "reverse",
        "flip_polarity",
        "crop",
        "dynamic_translation",
        "dynamic_rotation",
        "event_drop_3",
    ]
    acc = main(
        {"transforms": tran, "ssl_loss": "vicreg", "mode": "snn", "output_all": False}
    )

    tran = [
        "background_activity",
        "reverse",
        "flip_polarity",
        "crop",
        "transrot",
        "event_drop",
    ]
    acc = main(
        {"transforms": tran, "ssl_loss": "vicreg", "mode": "snn", "output_all": False}
    )

    tran = [
        "background_activity",
        "reverse",
        "flip_polarity",
        "crop",
        "transrot",
        "event_drop_2",
    ]
    acc = main(
        {"transforms": tran, "ssl_loss": "vicreg", "mode": "snn", "output_all": False}
    )

    tran = [
        "background_activity",
        "reverse",
        "flip_polarity",
        "crop",
        "transrot",
        "event_drop_3",
    ]
    acc = main(
        {"transforms": tran, "ssl_loss": "vicreg", "mode": "snn", "output_all": False}
    )

    # 3D-CNN
    tran = ["background_activity", "flip_polarity", "crop", "transrot"]
    acc = main(
        {"transforms": tran, "ssl_loss": "vicreg", "mode": "3dcnn", "output_all": False}
    )

    tran = ["background_activity", "flip_polarity", "crop", "transrot", "event_drop"]
    acc = main(
        {"transforms": tran, "ssl_loss": "vicreg", "mode": "3dcnn", "output_all": False}
    )

    tran = ["background_activity", "flip_polarity", "crop", "transrot", "event_drop_2"]
    acc = main(
        {"transforms": tran, "ssl_loss": "vicreg", "mode": "3dcnn", "output_all": False}
    )

    tran = ["background_activity", "flip_polarity", "crop", "transrot", "event_drop_3"]
    acc = main(
        {"transforms": tran, "ssl_loss": "vicreg", "mode": "3dcnn", "output_all": False}
    )

    tran = [
        "background_activity",
        "flip_polarity",
        "crop",
        "dynamic_translation",
        "dynamic_rotation",
        "event_drop_2",
    ]
    acc = main(
        {"transforms": tran, "ssl_loss": "vicreg", "mode": "3dcnn", "output_all": False}
    )

    tran = [
        "background_activity",
        "flip_polarity",
        "crop",
        "dynamic_translation",
        "dynamic_rotation",
        "event_drop_3",
    ]
    acc = main(
        {"transforms": tran, "ssl_loss": "vicreg", "mode": "3dcnn", "output_all": False}
    )

    # CNN
    tran = ["background_activity", "flip_polarity", "crop", "transrot"]
    acc = main(
        {"transforms": tran, "ssl_loss": "vicreg", "mode": "cnn", "output_all": False}
    )

    tran = ["background_activity", "flip_polarity", "crop", "transrot", "event_drop"]
    acc = main(
        {"transforms": tran, "ssl_loss": "vicreg", "mode": "cnn", "output_all": False}
    )

    tran = ["background_activity", "flip_polarity", "crop", "transrot", "event_drop_2"]
    acc = main(
        {"transforms": tran, "ssl_loss": "vicreg", "mode": "cnn", "output_all": False}
    )

    tran = ["background_activity", "flip_polarity", "crop", "transrot", "event_drop_3"]
    acc = main(
        {"transforms": tran, "ssl_loss": "vicreg", "mode": "cnn", "output_all": False}
    )

    tran = [
        "background_activity",
        "flip_polarity",
        "crop",
        "static_translation",
        "static_rotation",
        "event_drop_2",
    ]
    acc = main(
        {"transforms": tran, "ssl_loss": "vicreg", "mode": "cnn", "output_all": False}
    )

    tran = [
        "background_activity",
        "flip_polarity",
        "crop",
        "static_translation",
        "static_rotation",
        "event_drop_3",
    ]
    acc = main(
        {"transforms": tran, "ssl_loss": "vicreg", "mode": "cnn", "output_all": False}
    )

    exit()
    poss_trans = list(
        powerset(["flip", "background_activity", "reverse", "flip_polarity", "crop"])
    )
    print(poss_trans)
    best_acc = -2
    best_tran = None
    for curr in poss_trans:
        acc = main(
            {
                "transforms": list(curr),
                "ssl_loss": "vicreg",
                "mode": "3dcnn",
                "output_all": False,
            }
        )
        if acc > best_acc:
            best_acc = acc
            best_tran = list(curr)

    messss = f"BEST BASIC FOR CNN IS : {best_acc} {best_tran}"
    print(messss)
    report = open("report_3dcnn.txt", "a")
    report.write(f"{messss}\n\n\n")
    report.flush()
    report.close()

    # study based on transrot
    curr = [*best_tran, "static_translation", "static_rotation"]
    st = main(
        {"transforms": curr, "ssl_loss": "vicreg", "mode": "3dcnn", "output_all": False}
    )

    curr = [*best_tran, "dynamic_translation", "dynamic_rotation"]
    dyn = main(
        {"transforms": curr, "ssl_loss": "vicreg", "mode": "3dcnn", "output_all": False}
    )

    if dyn >= st:
        messss = f"BEST TRANSROT FOR CNN IS DYNAMIC = {dyn}"
        best_tran = [*best_tran, "dynamic_translation", "dynamic_rotation"]
    else:
        messss = f"BEST TRANSROT FOR CNN IS STATIC = {st}"
        best_tran = [*best_tran, "static_translation", "static_rotation"]

    print(messss)
    report = open("report_3dcnn.txt", "a")
    report.write(f"{messss}\n\n\n")
    report.flush()
    report.close()

    # study on cuts
    curr = [*best_tran, "cutout"]
    cutout = main(
        {"transforms": curr, "ssl_loss": "vicreg", "mode": "3dcnn", "output_all": False}
    )

    curr = [*best_tran, "event_drop"]
    eventdrop = main(
        {"transforms": curr, "ssl_loss": "vicreg", "mode": "3dcnn", "output_all": False}
    )

    curr = [*best_tran, "cutpaste"]
    cutpaste = main(
        {"transforms": curr, "ssl_loss": "vicreg", "mode": "3dcnn", "output_all": False}
    )

    curr = [*best_tran, "moving_occlusions"]
    movingocc = main(
        {"transforms": curr, "ssl_loss": "vicreg", "mode": "3dcnn", "output_all": False}
    )

    exit()

    # TODO: debug
    # trans = ['flip', 'background_activity', 'reverse', 'flip_polarity', 'event_drop']
    # main({'transforms': trans, 'ssl_loss': 'snn_loss_emd', 'mode':'snn'})

    # exit()
    trans = ["flip", "background_activity", "reverse", "flip_polarity", "event_drop"]
    main({"transforms": trans, "ssl_loss": "vicreg", "mode": "cnn"})

    trans = [
        "flip",
        "background_activity",
        "reverse",
        "flip_polarity",
        "dynamic_rotation",
        "dynamic_translation",
        "cutpaste",
        "moving_occlusion",
    ]
    main({"transforms": trans, "ssl_loss": "vicreg", "mode": "cnn"})

    trans = [
        "flip",
        "background_activity",
        "reverse",
        "flip_polarity",
        "crop",
        "dynamic_rotation",
        "dynamic_translation",
        "moving_occlusion",
    ]
    main({"transforms": trans, "ssl_loss": "vicreg", "mode": "cnn"})

    trans = [
        "flip",
        "background_activity",
        "reverse",
        "flip_polarity",
        "dynamic_rotation",
        "dynamic_translation",
        "event_drop",
    ]
    main({"transforms": trans, "ssl_loss": "vicreg", "mode": "cnn"})

    exit()

    # VIC

    # # exp - vicreg
    # trans = ['flip', 'background_activity', 'reverse', 'flip_polarity']
    # main({'transforms': trans, 'ssl_loss': 'vicreg'})

    # exp - try snn
    trans = ["flip", "background_activity", "reverse", "flip_polarity"]
    main({"transforms": trans, "ssl_loss": "barlow_twins", "mode": "cnn"})
    exit()

    # exp 2 (+crop)
    trans = ["flip", "background_activity", "reverse", "flip_polarity", "crop"]
    main({"transforms": trans})

    # exp 3 (+static rot/trans)
    trans = [
        "flip",
        "background_activity",
        "reverse",
        "flip_polarity",
        "static_translation",
        "static_rotation",
    ]
    main({"transforms": trans})

    # exp 4 (+cutout)
    trans = [
        "flip",
        "background_activity",
        "reverse",
        "flip_polarity",
        "static_translation",
        "static_rotation",
        "cutout",
    ]
    main({"transforms": trans})

    # exp 5 (+dyn - cutout)
    trans = [
        "flip",
        "background_activity",
        "reverse",
        "flip_polarity",
        "dynamic_translation",
        "dynamic_rotation",
    ]
    main({"transforms": trans})

    # exp 6 (+ dyn + movinOcc)
    trans = [
        "flip",
        "background_activity",
        "reverse",
        "flip_polarity",
        "dynamic_translation",
        "dynamic_rotation",
        "moving_occlusion",
    ]
    main({"transforms": trans})

    # static_trans = ['flip', 'background_activity', 'reverse',
    #                 'flip_polarity', 'static_rotation', 'static_translation', 'cutout']
    # dyn_trans = ['flip', 'background_activity', 'reverse', 'flip_polarity',
    #              'dynamic_rotation', 'dynamic_translation', 'moving_occlusion']

    # # static
    # main({
    #     'transforms': static_trans
    # })

    # for tr in static_trans:
    #     if tr == 'crop':
    #         continue

    #     new_tran = static_trans.copy()
    #     new_tran.remove(tr)
    #     main({
    #         'transforms': new_tran
    #     })

    # dynamic
    # main({
    #     'transforms': dyn_trans
    # })

    # for tr in dyn_trans:
    #     if tr == 'crop':
    #         continue

    #     new_tran = dyn_trans.copy()
    #     new_tran.remove(tr)
    #     main({
    #         'transforms': new_tran
    #     })

    # test
