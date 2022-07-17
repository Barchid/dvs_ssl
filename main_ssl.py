import pytorch_lightning as pl
from project.datamodules.dvs_datamodule import DVSDataModule
from project.utils.barlow_transforms import BarlowTwinsTransform
from project.ssl_module import SSLModule
import torch
import os
from matplotlib import pyplot as plt

from project.utils.eval_callback import OnlineFineTuner
# import traceback

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epochs = 1000
learning_rate = 1e-2 # barlowsnn=0.1, vicregsnn=0.01, dvs=1e-3
timesteps = 12
batch_size = 128
dataset = 'dvsgesture'
ssl_loss = 'barlow_twins'
output_all = False


def main(args):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="online_val_acc",  # TODO: select the logged metric to monitor the checkpoint saving
        filename="model-{epoch:03d}-{online_val_acc:.4f}",
        save_top_k=1,
        mode="max",
    )
    
    if 'mode' in args:
        mode = args['mode']
    else:
        mode = 'cnn'

    datamodule = DVSDataModule(
        batch_size,
        dataset,
        timesteps,
        data_dir='data',
        barlow_transf=args['transforms'],
        in_memory=False,
        num_workers=0,
        mode=mode
    )

    if 'ssl_loss' in args:
        ssl = args['ssl_loss']
    else:
        ssl = ssl_loss
        
    if 'lr' in args:
        lr = args['lr']
    else:
        lr=learning_rate

    module = SSLModule(
        n_classes=datamodule.num_classes,
        learning_rate=lr,
        epochs=epochs,
        ssl_loss=ssl,
        timesteps=timesteps,
        enc1=mode,
        enc2=mode,
        output_all=output_all
    )

    name = f"{dataset}_{ssl}"
    for tr in args['transforms']:
        name += f"_{tr}"

    online_finetuner = OnlineFineTuner(encoder_output_dim=512, num_classes=datamodule.num_classes, output_all=output_all)

    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=torch.cuda.device_count(),
        callbacks=[online_finetuner, checkpoint_callback],
        logger=pl.loggers.TensorBoardLogger("experiments", name=name),
        default_root_dir=f"experiments/{name}",
        precision=16
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
        report = open('errors.txt', 'a')
        report.write(f"{name} ===> error ! \n")
        report.flush()
        report.close()

    # write in score
    report = open('report.txt', 'a')
    report.write(f"{dataset} {name} {checkpoint_callback.best_model_score} \n")
    report.flush()
    report.close()


if __name__ == "__main__":
    pl.seed_everything(1234)

    trans = ['flip', 'background_activity', 'reverse', 'flip_polarity', 'crop', 'static_rotation', 'static_translation']
    main({'transforms': trans, 'ssl_loss': 'vicreg', 'mode':'snn'})
    
    trans = ['flip', 'background_activity', 'reverse', 'flip_polarity', 'event_drop']
    main({'transforms': trans, 'ssl_loss': 'vicreg', 'mode':'snn'})
    
    trans = ['flip', 'background_activity', 'reverse', 'flip_polarity', 'dynamic_rotation', 'dynamic_translation']
    main({'transforms': trans, 'ssl_loss': 'vicreg', 'mode':'snn'})
    
    trans = ['flip', 'background_activity', 'reverse', 'flip_polarity', 'dynamic_rotation', 'dynamic_translation', 'moving_occlusion']
    main({'transforms': trans, 'ssl_loss': 'vicreg', 'mode':'snn'})
    
    trans = ['flip', 'background_activity', 'reverse', 'flip_polarity', 'dynamic_rotation', 'dynamic_translation', 'cutpaste']
    main({'transforms': trans, 'ssl_loss': 'vicreg', 'mode':'snn'})
    
    trans = ['flip', 'background_activity', 'reverse', 'flip_polarity', 'dynamic_rotation', 'dynamic_translation', 'cutpaste', 'moving_occlusion']
    main({'transforms': trans, 'ssl_loss': 'vicreg', 'mode':'snn'})
    
    trans = ['flip', 'background_activity', 'reverse', 'flip_polarity', 'crop', 'dynamic_rotation', 'dynamic_translation', 'moving_occlusion']
    main({'transforms': trans, 'ssl_loss': 'vicreg', 'mode':'snn'})
    
    trans = ['flip', 'background_activity', 'reverse', 'flip_polarity', 'dynamic_rotation', 'dynamic_translation', 'event_drop']
    main({'transforms': trans, 'ssl_loss': 'vicreg', 'mode':'snn'})
    
    exit()
    
    # VIC

    # # exp - vicreg
    # trans = ['flip', 'background_activity', 'reverse', 'flip_polarity']
    # main({'transforms': trans, 'ssl_loss': 'vicreg'})

    # exp - try snn
    trans = ['flip', 'background_activity', 'reverse', 'flip_polarity']
    main({'transforms': trans, 'ssl_loss': 'barlow_twins', 'mode': 'snn'})
    exit()

    # exp 2 (+crop)
    trans = ['flip', 'background_activity', 'reverse', 'flip_polarity', 'crop']
    main({'transforms': trans})

    # exp 3 (+static rot/trans)
    trans = ['flip', 'background_activity', 'reverse', 'flip_polarity', 'static_translation', 'static_rotation']
    main({'transforms': trans})

    # exp 4 (+cutout)
    trans = ['flip', 'background_activity', 'reverse', 'flip_polarity',
             'static_translation', 'static_rotation', 'cutout']
    main({'transforms': trans})

    # exp 5 (+dyn - cutout)
    trans = ['flip', 'background_activity', 'reverse', 'flip_polarity', 'dynamic_translation', 'dynamic_rotation']
    main({'transforms': trans})

    # exp 6 (+ dyn + movinOcc)
    trans = ['flip', 'background_activity', 'reverse', 'flip_polarity',
             'dynamic_translation', 'dynamic_rotation', 'moving_occlusion']
    main({'transforms': trans})

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
