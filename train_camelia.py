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
        
parser = ArgumentParser(description='Training script for Camelia. If using a GPU (device argument), by default all GPUs will be used. To restrict GPU usage, in your command line: `export CUDA_VISIBLE_DEVICES=0` to select only the CUDA device 1 or e.g. `export CUDA_VISIBLE_DEVICES=0,1` to select CUDA device 0 and 1.' ,
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
dm_parse.add_argument('--neigh', type=boolean, default=True,
                      help='Whether to use neighboring methylation features.')
dm_parse.add_argument('--local', type=boolean, default=True,
                      help='Whether to use locally paired similarity feature.')
dm_parse.add_argument('--threshold', type=float, default=0.8,
                      help='Threshold on locally paired similarity feature.')
dm_parse.add_argument('--DNA', type=boolean, default=True,
                      help='Whether to use DNA context features.')

model_parse = parser.add_argument_group('Model', 'CaMelia model Hyperparameters')
model_parse.add_argument('--dropnans', type=boolean, default=False,
                         help='Whether to use samples that have an undefined locally paired similarity feature. If true, these will be dropped.')
model_parse.add_argument('--lr', type=float, default=0.1,
                         help='Learning rate.')
model_parse.add_argument('--max_depth', type=int, default=7,
                         help='Max depth of CatBoost trees.')
model_parse.add_argument('--verbose', type=int, default=100,
                         help='Verbosity level while learning.')
model_parse.add_argument('--eval_metric', type=str, default='AUC',
                         help='Metric to evaluate on.')
model_parse.add_argument('--device', type=str, default='CPU', choices=['CPU', 'GPU'],
                         help='Which computing device to use, either CPU or GPU.')


log_parse = parser.add_argument_group('Logging', 'Logging arguments')
log_parse.add_argument('--log_folder', type=str, default='logfolder',
                       help='Folder where the model checkpoints will be saved.')
log_parse.add_argument('--model_prefix', type=str, default='model_cell',
                       help='Prefix of the model names (one model for every cell). Models will be saved as "`model_prefix`00i.cbm" with i the cell index.')

args = parser.parse_args()

X = np.load(args.X)
y = np.load(args.y)
pos = np.load(args.pos)

from cpg_transformer.camelia import CaMeliaModel
from cpg_transformer.datamodules import CaMeliaPreprocessor

n_cells = y[list(y.keys())[0]].shape[1]


for i in range(n_cells):
    print('------ Processing & training Cell '+str(i)+' ------')
    preprocessor = CaMeliaPreprocessor(X, y, pos, val_keys=args.val_keys,
                                       test_keys=args.test_keys, fracs=args.fracs)
    preprocessor_outs = preprocessor(i, neigh=args.neigh, local=args.local,
                                     DNA=args.DNA, threshold=args.threshold)
    X_train, X_val, X_test, y_train, y_val, y_test, pos_val, pos_test = preprocessor_outs
    model = CaMeliaModel(dropnans=args.dropnans, learning_rate=args.lr, max_depth=args.max_depth,
                         verbose=args.verbose, eval_metric=args.eval_metric, device=args.device)
    model.fit(X_train, y_train)
    model.save(args.log_folder.rstrip('/')+'/'+args.model_prefix+f"{i:03d}.cbm")