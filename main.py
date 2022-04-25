from argparse import ArgumentParser
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from project.datamodules.dvs_datamodule import DVSDataModule
from project.dvs_module import DVSModule
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def main():
    # seeds the random from numpy, pytorch, etc for reproductibility
    pl.seed_everything(1234)

    args = get_args()

    datamodule = create_datamodule(args)

    module = create_module(args, datamodule)

    trainer = create_trainer(args)

    # Launch training/validation
    if args.mode == "train":
        trainer.fit(module, datamodule=datamodule, ckpt_path=args.ckpt_path)

        # report results in a txt file
        report_path = os.path.join('train_report.txt')
        report = open(report_path, 'a')

        # TODO: add any data you want to report here
        # here, we put the model's hyperparameters and the resulting val accuracy
        report.write(
            f"{args.name} {args.dataset} {args.event_representation} {args.blur_type} {args.learning_rate}  {trainer.checkpoint_callback.best_model_score}\n")
        report.flush()
    elif args.mode == "lr_find":
        lr_finder = trainer.tuner.lr_find(module, datamodule=datamodule)
        fig = lr_finder.plot(suggest=True)
        fig.show()
        print(f'SUGGESTION IS :', lr_finder.suggestion())
    elif args.mode == 'cam':
        datamodule.setup()
        val_loader = datamodule.val_dataloader()
        x, y = next(iter(val_loader))
        # x, y = x[0], y[0]
        target_layers = [module.model.layer4[-1]]
        input_tensor = x
        cam = GradCAM(model=module, target_layers=target_layers, use_cuda=torch.cuda.is_available())
        targets = [ClassifierOutputTarget(y[0])]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        
        print(grayscale_cam.shape, x[0].shape)
        
        exit()
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        plt.ims
        
        
    else:
        print(f'NOISE={args.noise} SEV={args.severity}')
        out = trainer.validate(module, datamodule=datamodule, ckpt_path=args.ckpt_path)
        metrics = out[0]
        report_path = os.path.join('experiments', 'validation_report.txt')
        report = open(report_path, 'a')
        report.write(f"{args.event_representation} {args.dataset} {args.noise} {args.severity} {metrics['val_acc']}\n")
        report.flush()
        report.close()


def create_module(args, datamodule: DVSDataModule) -> pl.LightningModule:
    # vars() is required to pass the arguments as parameters for the LightningModule
    dict_args = vars(args)
    if args.event_representation in ('frames_time', 'frames_event', 'bit_encoding'):
        dict_args['in_channels'] = args.timesteps * 2
    if args.event_representation == 'VoxelGrid':
        dict_args['in_channels'] = args.timesteps
    elif args.event_representation in ('weighted_frames', 'histogram', 'HOTS'):
        dict_args['in_channels'] = 2
        
    # dict_args['in_channels'] = 20 if args.event_representation in ('frames_event', 'frames_time') else 2
    # dict_args['in_channels'] = args.timesteps if args.event_representation == "VoxelGrid" else dict_args['in_channels']
    dict_args['num_classes'] = datamodule.num_classes

    # TODO: you can change the module class here
    module = DVSModule(**dict_args)

    return module


def create_datamodule(args) -> pl.LightningDataModule:
    # vars() is required to pass the arguments as parameters for the LightningDataModule
    dict_args = vars(args)

    # TODO: you can change the datamodule here
    datamodule = DVSDataModule(**dict_args)

    return datamodule


def create_trainer(args) -> pl.Trainer:
    # saves the best model checkpoint based on the accuracy in the validation set
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",  # TODO: select the logged metric to monitor the checkpoint saving
        filename="model-{epoch:03d}-{val_acc:.4f}",
        save_top_k=1,
        mode="max",
    )

    logger = pl.loggers.TensorBoardLogger(
        os.path.join(args.default_root_dir, "logger"),
        name=f"{args.name} {args.dataset} {args.event_representation}"
    )

    # create trainer
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback], logger=logger)
    return trainer


def get_args():
    # Program args
    # TODO: you can add program-specific arguments here
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, choices=["train", "validate", "lr_find", 'cam'], default="train")
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help="Path of a checkpoint file. Defaults to None, meaning the training/testing will start from scratch.")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataset', type=str, choices=["n-mnist",
                                                        "n-caltech101", "cifar10-dvs", "ncars", "asl-dvs", "dvsgesture"])
    parser.add_argument('--event_representation', type=str,
                        choices=["frames_time", "frames_event", "HATS", "HOTS", "VoxelGrid", "histogram", "weighted_frames", "bit_encoding"])
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--timesteps', type=int, default=8)
    parser.add_argument('--blur_type', type=str, choices=['averaging', 'gaussian', 'median', None], default=None)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--noise', type=str, choices=['hot_pixels', 'background_activity', 'occlusion', 'reverse', None], default=None)
    parser.add_argument('--severity', type=int, choices=[1,2,3,4,5, None], default=None)
    
    # Args for model
    parser = DVSModule.add_model_specific_args(parser)

    # Args for Trainer
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
