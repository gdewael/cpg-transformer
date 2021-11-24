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
        
parser = ArgumentParser(description='Training script for CpG Transformer.',
                        formatter_class=CustomFormatter)
parser.add_argument('X', type=str, metavar='X', help='NumPy file containing encoded genome.')
parser.add_argument('y', type=str, metavar='y', help='NumPy file containing methylation matrix.')
parser.add_argument('pos', type=str, metavar='pos', help='NumPy file containing positions of CpG sites.')

dm_parse = parser.add_argument_group('DataModule', 'Data Module arguments')
dm_parse.add_argument('--segment_size', type=int, default=1024,
                      help='Bin size in number of CpG sites (columns) that every batch will contain. If GPU memory is exceeded, this option can be lowered.')
dm_parse.add_argument('--fracs', type=float, nargs='+', default=[1,0,0],
                      help='Fraction of every chromosome that will go to train, val, test respectively. Is ignored for chromosomes that occur in --val_keys or --test_keys.')
dm_parse.add_argument('--mask_p', type=float, default=0.25,
                      help='How many sites to mask each batch as a percentage of the number of columns in the batch.')
dm_parse.add_argument('--mask_random_p', type=float, default=0.20,
                      help='The percentage of masked sites to instead randomize.')
dm_parse.add_argument('--resample_cells', type=int, default=None,
                      help='Whether to resample cells every training batch. Reduces complexity. If GPU memory is exceeded, this option can be used.')
dm_parse.add_argument('--resample_cells_val', type=int, default=None, 
                      help='Whether to resample cells every validation batch. If GPU memory is exceeded, this option can be used.')
dm_parse.add_argument('--val_keys', type=str, nargs='+', default=['chr5'],
                      help='Names/keys of validation chromosomes.')
dm_parse.add_argument('--test_keys', type=str, nargs='+', default=['chr10'], 
                      help='Names/keys of test chromosomes.')
dm_parse.add_argument('--batch_size', type=int, default=1,
                      help='Batch size.')
dm_parse.add_argument('--n_workers', type=int, default=4,
                      help='Number of worker threads to use in data loading. Increase if you experience a CPU bottleneck.')

model_parse = parser.add_argument_group('Model', 'CpG Transformer Hyperparameters')
model_parse.add_argument('--transfer_checkpoint', type=str, default=None,
                         help='.ckpt file to transfer model weights from. Has to be either a `.ckpt` pytorch lightning checkpoint or a `.pt` pytorch state_dict. If a `.ckpt` file is provided, then all following model arguments will not be used (apart from `--lr`). If a `.pt` file is provided, then all model arguments affecting the number of weights HAVE to correspond to those of the saved model. To perform transfer learning with models that have been trained on binary data and transfer them to continuous data (or vice versa), only .pt checkpoints can be used. When doing transfer learning, a lower-than-default learning rate (`--lr`) is advised.')
model_parse.add_argument('--RF', type=int, default=1001,
                         help='Receptive field of the underlying CNN.')
model_parse.add_argument('--n_conv_layers', type=int, default=2,
                         help='Number of convolutional layers, only 2 or 3 are possible.')
model_parse.add_argument('--DNA_embed_size', type=int, default=32,
                         help='Output embedding hidden size of the CNN.')
model_parse.add_argument('--cell_embed_size', type=int, default=32,
                         help='Cell embedding hidden size.')
model_parse.add_argument('--CpG_embed_size', type=int, default=32,
                         help='CpG embedding hidden size.')
model_parse.add_argument('--n_transformers', type=int, default=4,
                         help='Number of transformer modules to use.')
model_parse.add_argument('--act', type=str, default='relu',
                         help='Activation function in transformer feed-forward, either relu or gelu.')
model_parse.add_argument('--mode', type=str, choices=['2D', 'axial', 'intercell', 'intracell', 'none'], default='axial',
                         help='Attention mode.')
model_parse.add_argument('--transf_hsz', type=int, default=64,
                         help='Hidden dimension size in the transformer.')
model_parse.add_argument('--n_heads', type=int, default=8,
                         help='Number of self-attention heads.')
model_parse.add_argument('--head_dim', type=int, default=8,
                         help='Hidden dimensionality of each head.')
model_parse.add_argument('--window', type=int, default=21,
                         help='Window size of row-wise sliding window attention, should be odd.')
model_parse.add_argument('--layernorm', type=boolean, default=True,
                         help='Whether to apply layernorm in transformer modules.')
model_parse.add_argument('--CNN_do', type=float, default=.0,
                         help='Dropout rate in the CNN to embed DNA context.')
model_parse.add_argument('--transf_do', type=float, default=.2,
                         help='Dropout rate on the self-attention matrix.')
model_parse.add_argument('--lr', type=float, default=5e-4,
                         help='Learning rate.')
model_parse.add_argument('--lr_decay_factor', type=float, default=.90,
                         help='Learning rate multiplicative decay applied after every epoch.')
model_parse.add_argument('--warmup_steps', type=int, default=1000,
                         help='Number of steps over which the learning rate will linearly warm up.')

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

if np.all(np.mod(y[list(y.keys())[0]], 1) == 0):
    data_mode = 'binary'
else:
    data_mode = 'continuous'

from cpg_transformer.cpgtransformer import CpGTransformer
from cpg_transformer.datamodules import CpGTransformerDataModule

n_cells = y[list(y.keys())[0]].shape[1]


if args.transfer_checkpoint:
    assert args.transfer_checkpoint.endswith('.ckpt') or args.transfer_checkpoint.endswith('.pt'), 'Pretrained models should be a .ckpt or .pt file'
    if args.transfer_checkpoint.endswith('.ckpt'):
        model = CpGTransformer.load_from_checkpoint(args.transfer_checkpoint, lr=args.lr)
        model.cell_embed = torch.nn.Embedding(n_cells, model.hparams.cell_embed_size)
    else:
        pretrained_model_state = torch.load(args.transfer_checkpoint)
        n_cells_pretrained = pretrained_model_state['cell_embed.weight'].shape[0]
        model = CpGTransformer(n_cells_pretrained, RF=args.RF, n_conv_layers=args.n_conv_layers,
                       CNN_do=args.CNN_do, data_mode = data_mode,
                       DNA_embed_size=args.DNA_embed_size, cell_embed_size=args.cell_embed_size,
                       CpG_embed_size=args.CpG_embed_size, transf_hsz=args.transf_hsz,
                       transf_do=args.transf_do, act=args.act, n_transformers=args.n_transformers,
                       n_heads=args.n_heads, head_dim=args.head_dim, window=args.window,
                       layernorm=args.layernorm, lr=args.lr, lr_decay_factor=args.lr_decay_factor,
                       warmup_steps=args.warmup_steps, mode=args.mode)
        model.load_state_dict(pretrained_model_state, strict = False)
        model.cell_embed = torch.nn.Embedding(n_cells, model.hparams.cell_embed_size)       
        model.hparams.n_cells = n_cells
        
else:
    model = CpGTransformer(n_cells, RF=args.RF, n_conv_layers=args.n_conv_layers, CNN_do=args.CNN_do,
                       DNA_embed_size=args.DNA_embed_size, cell_embed_size=args.cell_embed_size,
                       CpG_embed_size=args.CpG_embed_size, transf_hsz=args.transf_hsz,
                       transf_do=args.transf_do, act=args.act, n_transformers=args.n_transformers,
                       n_heads=args.n_heads, head_dim=args.head_dim, window=args.window,
                       layernorm=args.layernorm, lr=args.lr, lr_decay_factor=args.lr_decay_factor,
                       warmup_steps=args.warmup_steps, mode=args.mode, data_mode = data_mode)
    
    

datamodule = CpGTransformerDataModule(X, y, pos, segment_size=args.segment_size, fracs=args.fracs,
                                      RF=model.RF, mask_perc=args.mask_p, mask_random_perc=args.mask_random_p,
                                      resample_cells=args.resample_cells,
                                      resample_cells_val=args.resample_cells_val,
                                      val_keys=args.val_keys, test_keys=args.test_keys, 
                                      batch_size=args.batch_size, n_workers=args.n_workers)

print('Running in', model.hparams.data_mode, 'mode.')
callbacks = [ModelCheckpoint(monitor='val_loss', mode='min')]
if args.tensorboard:
    logger = TensorBoardLogger(args.log_folder, name=args.experiment_name)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks += [lr_monitor]
if args.earlystop:
    earlystopper = EarlyStopping(monitor='val_loss',patience=args.patience,mode='min')
    callbacks += [earlystopper]

trainer = Trainer.from_argparse_args(args, logger=logger, callbacks=callbacks)

trainer.fit(model, datamodule)