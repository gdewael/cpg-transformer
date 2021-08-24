import numpy as np
from argparse import ArgumentParser
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def boolean(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

        
class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.MetavarTypeHelpFormatter):
    pass
        
parser = ArgumentParser(description='Training script for DeepCpG.',
                        formatter_class=CustomFormatter)
parser.add_argument('X', type=str, metavar='X', help='NumPy file containing encoded genome.')
parser.add_argument('y', type=str, metavar='y', help='NumPy file containing methylation matrix.')
parser.add_argument('pos', type=str, metavar='pos', help='NumPy file containing positions of CpG sites.')

dm_parse = parser.add_argument_group('DataModule', 'Data Module arguments')
dm_parse.add_argument('--fracs', type=float, nargs='+', default=[1,0,0],
                      help='Fraction of every chromosome that will go to train, val, test respectively. Is ignored for chromosomes that occur in --val_keys or --test_keys.')
dm_parse.add_argument('--val_keys', type=str, nargs='+', default=['chr5'],
                      help='Names/keys of validation chromosomes.')
dm_parse.add_argument('--test_keys', type=str, nargs='+', default=['chr10'], 
                      help='Names/keys of test chromosomes.')
dm_parse.add_argument('--batch_size', type=int, default=128,
                      help='Batch size.')
dm_parse.add_argument('--n_workers', type=int, default=4,
                      help='Number of worker threads to use in data loading. Increase if you experience a CPU bottleneck.')
dm_parse.add_argument('--window', type=int, default=25,
                      help='How many sites to use up and downstream to make up local context in RNN.')
dm_parse.add_argument('--max_dist', type=int, default=25000,
                      help='Maximal distance over which relative distances are normalized. CpG sites farther away will get assigned the max relative distance (1). ')

model_parse = parser.add_argument_group('Model', 'DeepCpG model Hyperparameters')
model_parse.add_argument('--RF', type=int, default=1001,
                         help='Receptive field of the underlying CNN.')
model_parse.add_argument('--n_conv_layers', type=int, default=2,
                         help='Number of convolutional layers, only 2 or 3 are possible.')
model_parse.add_argument('--do_CNN', type=float, default=.0,
                         help='Dropout rate in the CNN.')
model_parse.add_argument('--do_RNN', type=float, default=.0,
                         help='Dropout rate in the RNN.')
model_parse.add_argument('--do_joint1', type=float, default=.0,
                         help='Dropout rate in the first layer of the Joint module.')
model_parse.add_argument('--do_joint2', type=float, default=.0,
                         help='Dropout rate in the second layer of the Joint module.')
model_parse.add_argument('--lr', type=float, default=1e-4,
                         help='Learning rate.')
model_parse.add_argument('--lr_decay_factor', type=float, default=.95,
                         help='Learning rate multiplicative decay applied after every epoch.')
model_parse.add_argument('--warmup_steps', type=int, default=1000,
                         help='Number of steps over which the learning rate will linearly warm up.')
model_parse.add_argument('--e2e', type=boolean, default=True,
                         help='Whether to train end-to-end, or optimize CNN, CpG and joint module separately.')
model_parse.add_argument('--CNN_epochs', type=int, default=50,
                         help='If not using e2e training, how much epochs to train the CNN for. Overrides the max_epochs Trainer argument.')
model_parse.add_argument('--RNN_epochs', type=int, default=50,
                        help='If not using e2e training, how much epochs to train the RNN for. Overrides the max_epochs Trainer argument.')
model_parse.add_argument('--joint_epochs', type=int, default=25,
                        help='If not using e2e training, how much epochs to train the joint for. Overrides the max_epochs Trainer argument.')
model.parse.add_argument('--lr_factor_joint', type=float, default=0.05,
                         help='Factor with which to multiply the learning rate when training the joint module. Empirically, we found that a high learning rate for optimizing the joint module only leads to overfitting.')


log_parse = parser.add_argument_group('Logging', 'Logging arguments')
log_parse.add_argument('--tensorboard', type=boolean, default=True,
                       help='Whether to use tensorboard. If True, then training progress can be followed by using (1) `tensorboard --logdir logfolder/` in a separate terminal and (2) accessing at localhost:6006.')
log_parse.add_argument('--log_folder', type=str, default='logfolder',
                       help='Folder where the tensorboard logs will be saved. Will additinally contain saved model checkpoints.')
log_parse.add_argument('--experiment_name', type=str, default='experiment',
                       help='Name of the run within the log folder.')
log_parse.add_argument('--earlystop', type=boolean, default=True,
                       help='Whether to use early stopping after the validation loss has not decreased for `patience` epochs.')
log_parse.add_argument('--patience', type=int, default=10,
                       help='Number of epochs to wait for a possible decrease in validation loss before early stopping.')


parser = Trainer.add_argparse_args(parser)

args = parser.parse_args()

X = np.load(args.X)
y = np.load(args.y)
pos = np.load(args.pos)

from cpg_transformer.deepcpg import DeepCpG
from cpg_transformer.datamodules import DeepCpGDataModule

n_cells = y[list(y.keys())[0]].shape[1]



model = DeepCpG(n_cells, n_conv_layers=args.n_conv_layers, do_CNN=args.do_CNN, RF=args.RF,
                 do_RNN=args.do_RNN, do_joint1=args.do_joint1, do_joint2=args.do_joint2,
                 lr=args.lr, warmup_steps=args.warmup_steps, lr_decay_factor=args.lr_decay_factor)

datamodule = DeepCpGDataModule(X, y, pos, RF=model.RF, fracs=args.fracs,
                 val_keys=args.val_keys, test_keys=args.test_keys,
                 batch_size=args.batch_size, n_workers=args.n_workers,
                 window=args.window, max_dist=args.max_dist, batch_pos=False)


callbacks = [ModelCheckpoint(monitor='val_loss', mode='min')]
if args.tensorboard:
    logger = TensorBoardLogger(args.logfolder, name=args.experiment_name)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks += [lr_monitor]
if args.earlystop:
    earlystopper = EarlyStopping(monitor='val_loss',patience=args.patience,mode='min')
    callbacks += [earlystopper]



if args.e2e:
    trainer = Trainer.from_argparse_args(args, logger=logger, callbacks=callbacks)
    trainer.fit(model, datamodule)
    
else:
    model.forward = model.forward_CNN
    model.hparams.lr = args.lr
    print('----TRAINING CNN MODULE-----')
    trainer = Trainer.from_argparse_args(args, logger=logger, callbacks=callbacks, max_epochs=args.CNN_epochs)
    trainer.fit(model, datamodule)
    
    model.forward = model.forward_RNN
    model.hparams.lr = args.lr
    print('----TRAINING RNN MODULE-----')
    trainer = Trainer.from_argparse_args(args, logger=logger, callbacks=callbacks, max_epochs=args.RNN_epochs)
    trainer.fit(model, datamodule)
    
    model.forward = model.forward_joint
    model.hparams.lr = args.lr*args.lr_factor_joint
    print('----TRAINING JOINT MODULE-----')
    trainer = Trainer.from_argparse_args(args, logger=logger, callbacks=callbacks, max_epochs=args.joint_epochs)
    trainer.fit(model, datamodule)