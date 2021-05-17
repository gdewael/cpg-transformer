import os
import numpy as np
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


parser = ArgumentParser()
parser.add_argument('--X', type=str)
parser.add_argument('--y', type=str)
parser.add_argument('--pos', type=str)

parser.add_argument('--segment_size', type=int, default=1250)
parser.add_argument('--maxlen', type=int, default=500*1250)
parser.add_argument('--mask_gaps', type=utils.str2bool, default=True)
parser.add_argument('--fracs', type=float, nargs='+', default=[1,0,0])
parser.add_argument('--val_keys', type=str, nargs='+', default=['chr5+', 'chr5-'])
parser.add_argument('--test_keys', type=str, nargs='+', default=['chr10+','chr10-'])

parser.add_argument('--tensorboard_name', type=str, default='MethylationCNN')
parser.add_argument('--tensorboard_logs', type=str, default='./logs')
parser.add_argument('--multi_gpu', type=utils.str2bool, default=False)

args = parser.parse_args()

X = np.load(args.X)
y = np.load(args.y)
pos = np.load(args.pos)





from DataModules import *
from CpGTransformer import *

#model = SCellCpGTransformerDeepCpG(RF=1001, n_conv_layers=2, do_CNN=0, CNN_out_size=32,
#                 n_cells = 122, cell_embed_size=32, CpG_embed_size=32, transf_hsz=64,
#                 dropout=0.20, act='relu', weight_factor=None,
#                 n_transformers=4, n_heads=8, head_dim=8, window=25, val_window=25,
#                                   tflayernorm=True,
#                 lr=5e-4, lr_decay_factor=.90, warmup_steps=1000)

model = SCellCpGTransformerDeepCpG.load_from_checkpoint('logs/hemato_S_w25_c16/version_0/checkpoints/epoch=0-step=49.ckpt')


datamodule = CpGTransformerDataModule(X, y, pos, cov, segment_size=1024, RF=model.RF,
             maxlen=None, fracs=[1,0,0],
             windowed=True, exclude_centerpos=False,
             singlecell=True, mask_perc=0.10, mask_random_perc=0.2,
             val_keys=['chr5'], test_keys=['chr10'],
             batch_size=1, n_workers=4, resample_cells=16, resample_cells_val=16)

logger = TensorBoardLogger('logs/', name="hemato_S_w25_c16")
lr_monitor = LearningRateMonitor(logging_interval='step')
checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min')

trainer = Trainer(gpus=2, accelerator='ddp', logger=logger, min_epochs=5, max_epochs=100,
                  #val_check_interval=50, limit_val_batches=10,
                  callbacks=[lr_monitor, checkpoint_callback],terminate_on_nan=True, enable_pl_optimizer=True)

trainer.fit(model, datamodule)