import pytorch_lightning as pl
from project.utils.rgb_ssl_transforms import BarlowTwinsTransform, cifar10_normalization
from project.ssl_module import SSLModule
import torch
import os
from matplotlib import pyplot as plt
from torchvision.datasets import CIFAR10
from project.utils.eval_callback import OnlineFineTuner
from torch.utils.data import DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epochs = 200
learning_rate = 2e-4
timesteps = 1.5
batch_size = 128
dataset = 'cifar10-dvs'
ssl_loss = 'barlow_twins'


def main():
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_eval_acc",  # TODO: select the logged metric to monitor the checkpoint saving
        filename="model-{epoch:03d}-{val_eval_acc:.4f}",
        save_top_k=1,
        mode="max",
    )

    train_trans = BarlowTwinsTransform(normalize=cifar10_normalization())
    train_set = CIFAR10(root="data/", train=True, download=True, transform=train_trans)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    val_trans = BarlowTwinsTransform(normalize=cifar10_normalization())
    val_set = CIFAR10(root="data/", train=False, download=True, transform=val_trans)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

    module = SSLModule(
        n_classes=10,
        learning_rate=learning_rate,
        epochs=epochs,
        ssl_loss=ssl_loss,
        timesteps=timesteps
    )

    name = f"rgb_{ssl_loss}"

    online_finetuner = OnlineFineTuner(encoder_output_dim=512 * 3, num_classes=10)

    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=torch.cuda.device_count(),
        callbacks=[online_finetuner, checkpoint_callback],
        logger=pl.loggers.TensorBoardLogger("experiments", name=name),
        default_root_dir=f"experiments/{name}"
    )

    lr_finder = trainer.tuner.lr_find(module, train_loader, val_loader)
    fig = lr_finder.plot(suggest=True)
    fig.savefig('lr.png')   # save the figure to file
    plt.close(fig)    # close th
    print(f'SUGGESTION IS :', lr_finder.suggestion())
    exit()
    # trainer.fit(module, train_loader, val_loader)

    # write in score
    report = open('report.txt', 'a')
    report.write(f"{dataset} {name} {checkpoint_callback.best_model_score} \n")
    report.flush()
    report.close()


if __name__ == "__main__":
    pl.seed_everything(1234)

    # static
    main()
