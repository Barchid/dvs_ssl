import pytorch_lightning as pl
from project.datamodules.dvs_datamodule import DVSDataModule
from project.utils.barlow_transforms import BarlowTwinsTransform
from project.ssl_module import SSLModule
import torch
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epochs = 150
learning_rate = 1e-3
timesteps = 12
batch_size = 32
dataset = 'n-mnist'
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
        ssl_loss=ssl_loss
    )
    
    name = ""
    for tr in args['transforms']:
        name += f"_{tr}"
    
    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=torch.cuda.device_count(),
        callbacks=[checkpoint_callback],
        logger=pl.loggers.TensorBoardLogger("experiments", name=name)
    )
    
    trainer.fit(module, datamodule=datamodule)
    
    # write in score    
    report = open('report.txt', 'a')
    report.write(f"{dataset} {name} {checkpoint_callback.best_model_score} \n")
    report.flush()
    report.close()

if __name__ == "__main__":
    static_trans = ['flip', 'background_activity', 'reverse', 'flip_polarity', 'crop', 'static_rotation', 'static_translation']
    dyn_trans = ['flip', 'background_activity', 'reverse', 'flip_polarity', 'crop', 'dynamic_rotation', 'dynamic_translation']
    
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
