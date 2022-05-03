import pytorch_lightning as pl
from project.datamodules.dvs_datamodule import DVSDataModule
from project.utils.barlow_transforms import BarlowTwinsTransform
from project.ssl_module import SSLModule
import torch
import os
from matplotlib import pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epochs = 200
learning_rate = 2e-4
timesteps = 32
batch_size = 32
dataset = 'cifar10-dvs'
ssl_loss = 'barlow_twins'

def main(args):    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_eval_acc",  # TODO: select the logged metric to monitor the checkpoint saving
        filename="model-{epoch:03d}-{val_eval_acc:.4f}",
        save_top_k=1,
        mode="max",
    )
    
    datamodule = DVSDataModule(batch_size, dataset, timesteps, data_dir='data', barlow_transf=args['transforms'])
    
    module = SSLModule(
        n_classes=datamodule.num_classes,
        learning_rate=learning_rate,
        epochs=epochs,
        ssl_loss=ssl_loss,
        timesteps=timesteps
    )
    
    name = ""
    for tr in args['transforms']:
        name += f"_{tr}"
    
    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=torch.cuda.device_count(),
        callbacks=[checkpoint_callback],
        logger=pl.loggers.TensorBoardLogger("experiments", name=name),
        default_root_dir=f"experiments/{name}"
    )
    
    lr_finder = trainer.tuner.lr_find(module, datamodule=datamodule)
    fig = lr_finder.plot(suggest=True)
    fig.savefig('lr.png')   # save the figure to file
    plt.close(fig)    # close th
    print(f'SUGGESTION IS :', lr_finder.suggestion())
    exit()
    # trainer.fit(module, datamodule=datamodule)
    
    # write in score    
    report = open('report.txt', 'a')
    report.write(f"{dataset} {name} {checkpoint_callback.best_model_score} \n")
    report.flush()
    report.close()

if __name__ == "__main__":
    static_trans = ['flip', 'background_activity', 'reverse', 'flip_polarity', 'crop', 'static_rotation', 'static_translation', 'cutout']
    dyn_trans = ['flip', 'background_activity', 'reverse', 'flip_polarity', 'crop', 'dynamic_rotation', 'dynamic_translation', 'moving_occlusion']
    
    # static
    main({
        'transforms': static_trans
    })
    
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
