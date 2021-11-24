# CpG Transformer

This repository contains code, pre-trained models and instructions on how to use CpG Transformer ([published paper link](https://doi.org/10.1093/bioinformatics/btab746))
for imputation of single-cell methylomes.

**New from 24-11-2021 onwards:** CpG Transformer now supports continuous methylation calls as input, for which it will train a regression model in the same fashion as described in our publication, but then with the mean-squared error as loss function. Performances for regression have not been benchmarked as of yet.

<details><summary>Table of contents</summary>
  
- [Comparison of single-cell methylome imputation performance](#perf-comp)
- [Installation](#install)
- [Usage](#usage)
  - [Quick Start](#quickstart)
  - [Input formatting](#input)
  - [Training](#train)
  - [Imputation and denoising](#impute)
  - [Benchmarking](#benchmark)
  - [Interpretation](#interpret)
- [Pre-trained models](#pretrained)
- [Citation](#citation)
- [License](#license)
</details>


## Comparison of single-cell methylome imputation performance <a name="perf-comp"></a>

| Dataset | # cells | ROC AUC [DeepCpG](https://doi.org/10.1186/s13059-017-1189-z) \* | ROC AUC [CaMelia](https://doi.org/10.1093/bioinformatics/btab029) \* | ROC AUC CpG Transformer |
| - | - | - | - | - | 
| [Ser](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE56879) | 20 | 90.21 | 90.22 | **91.55** | 
| [2i](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE56879) | 12 | 84.80 | 83.02 | **85.77** | 
| [HCC](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE65364) | 25 | 96.89 | 97.42 | **97.96** | 
| [MBL](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE125499) | 30 | 88.22 | 89.17 | **92.49** |
| [Hemato](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE87197) | 122 | 88.85 | 89.16 | **90.65** |


\* Results obtained with reproduced, optimized code, also found in this repository.


## Installation <a name="install"></a>

CpG Transformer is implemented in [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning).

If you have one or more GPUs on your machine, we recommend running CpG Transformer locally using a conda environment.
Make sure to install the correct version of [PyTorch](https://pytorch.org/get-started/locally/) (using the cuda version that is installed on your system).
The following shows an example installation process for a system running CUDA 11.1:

```bash
conda create --name cpgtransformer
source activate cpgtransformer
conda install pip
pip install torch
pip install pytorch-lightning biopython pandas numpy
git clone https://github.com/gdewael/cpg-transformer.git
```

In case CpG Transformer loses backwards compatibility with more-recent versions of PyTorch and PyTorch Lightning: this repo has been tested with up to Python 3.9, PyTorch 1.10, PyTorch Lightning 1.5

For CaMelia training and imputation, additionally do:
```bash
pip install catboost
```

If your machine does not have a GPU, we provide Google Colab transfer learning and imputation notebooks that run on Google cloud resources. (see [Quick Start](#quickstart)).


## Usage <a name="usage"></a>

### Quick Start  <a name="quickstart"></a>


To quickly test out CpG Transformer, we provide Google Drive access to the preprocessed files for the Ser dataset, which can be downloaded [here](https://drive.google.com/drive/folders/1zNvyOX0F0ztDFEsgwaeTdsxJYo0_fQgg).

```bash
# Train a CpG Transformer model
python train_cpg_transformer.py X_ser.npz y_ser.npz pos_ser.npz --gpus 1 # train from scratch with one gpu
python train_cpg_transformer.py X_ser.npz y_ser.npz pos_ser.npz --gpus 2 --accelerator ddp # train with multiple gpus
python train_cpg_transformer.py X_ser.npz y_ser.npz pos_ser.npz --gpus 1 --transfer_checkpoint data/model_checkpoints/Ser_model.pt # transfer learning

# Impute a dataset with a trained model
python impute_genome.py cpg_transformer X_ser.npz y_ser.npz pos_ser.npz output_ser.npz --model_checkpoint path/to/saved/model.ckpt
```

We additionally provide Google Colab notebooks for those with no local GPU resources:
- Training: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gdewael/cpg-transformer/blob/main/notebooks/train_cpg_transformer.ipynb)
- Imputation: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gdewael/cpg-transformer/blob/main/notebooks/impute_cpg_transformer.ipynb)


### Input formatting  <a name="input"></a>

CpG Transformer uses NumPy `.npz` zipped archive files as inputs. More specifically, 3 input files are necessary:

**(1)** `X.npz`. An encoded genome as a dictionary of NumPy arrays. Every key-value pair corresponds to a chromosome, with the key the name of the chromosome (e.g.: `'chr1'`) and the value a 1D NumPy array of encoded sequence (e.g.: `np.array([0,2,2,3,...,1,1,2])`). Sequences are encoded according to:
```
{'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4,
'M': 5, 'R': 6, 'W': 7, 'S': 8, 'Y': 9,
'K': 10, 'V': 11, 'H': 12, 'D': 13,
'B': 14, 'X': 15}
```

**(2)** `y.npz`. A partially observed methylation matrix as a dictionary of NumPy arrays. Every key-value pair corresponds to a chromosome, with the key the name of the chromosome (e.g.: `'chr1'`) and the value a 2D NumPy array corresponding to the methylation matrix for that chromosome. Every methylation matrix is a `# sites * # cells` matrix with every element at row `i` and column `j` denoting the methylation state of CpG site `i` of cell `j`. Methylation states are encoded by `-1 = unknown`, `0 = unmethylated`, `1 = methylated`. For training, we recommend only including CpG sites where at least one cell has an observed state, as columns without observation confer no useful information when training. Note that CpG Transformer only accepts forwards strand methylation states. If your methylation calls are recorded for both strands separately, you should combine them to the forward strand. **New:** for continuous methylation calls, methylation states should be encoded as values `-1 = unknown`, or `[0...1] = methylation frequency`.

**(3)** `pos.npz`. Positions (0-indexed) of all input CpG sites as a dictionary of NumPy arrays. Every key-value pair corresponds to a chromosome, with the key the name of the chromosome (e.g.: `'chr1'`) and the value a 1D NumPy array corresponding to the locations of all profiled CpG sites in that chromosome (columns in the second input).

Example:

```python
>>> X['chr1'].shape
(197195432,)
>>> X['chr1']
array([4, 4, 4, ..., 1, 1, 2], dtype=int8)

>>> y['chr1'].shape
(1190072, 20)
>>> y['chr1']
array([[-1,  1, -1, ..., -1,  1, -1],
       [-1,  1, -1, ...,  1, -1, -1],
       [-1,  0, -1, ...,  0, -1, -1],
       ...,
       [-1, -1, -1, ..., -1, -1, -1],
       [-1, -1, -1, ..., -1, -1, -1],
       [-1, -1, -1, ..., -1, -1, -1]], dtype=int8)

>>> pos['chr1'].shape
(1190072,)
>>> pos['chr1']
array([  3000573,   3000725,   3000900, ..., 197194914, 197194986,
       197195054], dtype=int32)
```

For the datasets used in our paper, we provide template preprocessing scripts in the `data` folder, along with [instructions](https://github.com/gdewael/cpg-transformer/tree/main/data#readme) on where to download all relevant data. In addition, we provide a simple script for .tsv file inputs. (see the [README](https://github.com/gdewael/cpg-transformer/tree/main/data#readme)).


### Training <a name="train"></a>

Separate training scripts are provided for training CpG Transformer models (`train_cpg_transformer.py`), DeepCpG (`train_deepcpg.py`) and CaMelia (`train_camelia.py`). In the following, only the arguments to CpG Transformer will be shown. For all scripts, arguments and their explanations can be accessed with the `-h` help flag.

Arguments to CpG Transformer are split into 4 groups: (1) DataModule arguments concern how the data will be preprocessed and loaded for the model to use, (2) Model arguments concern model architecture and training, (3) Logging arguments determine how the training process can be followed and where model weights will be saved and (4) pl.Trainer arguments list all arguments to the PyTorch Lightning trainer object. Most of these arguments are not applicable to standard use of CpG Transformer but are kept in for full flexibility. For more information on pl.Trainer we refer its [documentation](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#).


<details><summary>All train_cpg_transformer.py flags</summary>

```
usage: train_cpg_transformer.py [-h] [--segment_size int] [--fracs float [float ...]] [--mask_p float]
                                [--mask_random_p float] [--resample_cells int] [--resample_cells_val int]
                                [--val_keys str [str ...]] [--test_keys str [str ...]] [--batch_size int]
                                [--n_workers int] [--transfer_checkpoint str] [--RF int]
                                [--n_conv_layers int] [--DNA_embed_size int] [--cell_embed_size int]
                                [--CpG_embed_size int] [--n_transformers int] [--act str]
                                [--mode {2D,axial,intercell,intracell,none}] [--transf_hsz int]
                                [--n_heads int] [--head_dim int] [--window int] [--layernorm boolean]
                                [--CNN_do float] [--transf_do float] [--lr float] [--lr_decay_factor float]
                                [--warmup_steps int] [--tensorboard boolean] [--log_folder str]
                                [--experiment_name str] [--earlystop boolean] [--patience int]
                                [--logger [str_to_bool]] [--checkpoint_callback [str_to_bool]]
                                [--default_root_dir str] [--gradient_clip_val float]
                                [--gradient_clip_algorithm str] [--process_position int] [--num_nodes int]
                                [--num_processes int] [--gpus _gpus_allowed_type]
                                [--auto_select_gpus [str_to_bool]] [--tpu_cores _gpus_allowed_type]
                                [--log_gpu_memory str] [--progress_bar_refresh_rate int]
                                [--overfit_batches _int_or_float_type] [--track_grad_norm float]
                                [--check_val_every_n_epoch int] [--fast_dev_run [str_to_bool_or_int]]
                                [--accumulate_grad_batches int] [--max_epochs int] [--min_epochs int]
                                [--max_steps int] [--min_steps int] [--max_time str]
                                [--limit_train_batches _int_or_float_type]
                                [--limit_val_batches _int_or_float_type]
                                [--limit_test_batches _int_or_float_type]
                                [--limit_predict_batches _int_or_float_type]
                                [--val_check_interval _int_or_float_type] [--flush_logs_every_n_steps int]
                                [--log_every_n_steps int] [--accelerator str]
                                [--sync_batchnorm [str_to_bool]] [--precision int] [--weights_summary str]
                                [--weights_save_path str] [--num_sanity_val_steps int]
                                [--truncated_bptt_steps int] [--resume_from_checkpoint str] [--profiler str]
                                [--benchmark [str_to_bool]] [--deterministic [str_to_bool]]
                                [--reload_dataloaders_every_epoch [str_to_bool]]
                                [--auto_lr_find [str_to_bool_or_str]] [--replace_sampler_ddp [str_to_bool]]
                                [--terminate_on_nan [str_to_bool]]
                                [--auto_scale_batch_size [str_to_bool_or_str]]
                                [--prepare_data_per_node [str_to_bool]] [--plugins str] [--amp_backend str]
                                [--amp_level str] [--distributed_backend str]
                                [--move_metrics_to_cpu [str_to_bool]] [--multiple_trainloader_mode str]
                                [--stochastic_weight_avg [str_to_bool]]
                                X y pos

Training script for CpG Transformer.

positional arguments:
  X                     NumPy file containing encoded genome.
  y                     NumPy file containing methylation matrix.
  pos                   NumPy file containing positions of CpG sites.

optional arguments:
  -h, --help            show this help message and exit
```

</details>

<details><summary>DataModule arguments</summary>
    
```
DataModule:
  Data Module arguments

  --segment_size int    Bin size in number of CpG sites (columns) that every batch will contain.
                        If GPU memory is exceeded, this option can be lowered. (default: 1024)
  --fracs float [float ...]
                        Fraction of every chromosome that will go to train, val, test
                        respectively. Is ignored for chromosomes that occur in --val_keys or
                        --test_keys. (default: [1, 0, 0])
  --mask_p float        How many sites to mask each batch as a percentage of the number of
                        columns in the batch. (default: 0.25)
  --mask_random_p float
                        The percentage of masked sites to instead randomize. (default: 0.2)
  --resample_cells int  Whether to resample cells every training batch. Reduces complexity. If
                        GPU memory is exceeded, this option can be used. (default: None)
  --resample_cells_val int
                        Whether to resample cells every validation batch. If GPU memory is
                        exceeded, this option can be used. (default: None)
  --val_keys str [str ...]
                        Names/keys of validation chromosomes. (default: ['chr5'])
  --test_keys str [str ...]
                        Names/keys of test chromosomes. (default: ['chr10'])
  --batch_size int      Batch size. (default: 1)
  --n_workers int       Number of worker threads to use in data loading. Increase if you
                        experience a CPU bottleneck. (default: 4)
```
    
</details>

<details><summary>Model arguments</summary>
    
```
Model:
  CpG Transformer Hyperparameters

  --transfer_checkpoint str
                        .ckpt file to transfer model weights from. Has to be either a `.ckpt`
                        pytorch lightning checkpoint or a `.pt` pytorch state_dict. If a `.ckpt`
                        file is provided, then all following model arguments will not be used
                        (apart from `--lr`). If a `.pt` file is provided, then all following
                        model arguments HAVE to correspond to the arguments of the saved model.
                        When doing transfer learning, a lower-than-default learning rate (`--lr`)
                        is advised. (default: None)
  --RF int              Receptive field of the underlying CNN. (default: 1001)
  --n_conv_layers int   Number of convolutional layers, only 2 or 3 are possible. (default: 2)
  --DNA_embed_size int  Output embedding hidden size of the CNN. (default: 32)
  --cell_embed_size int
                        Cell embedding hidden size. (default: 32)
  --CpG_embed_size int  CpG embedding hidden size. (default: 32)
  --n_transformers int  Number of transformer modules to use. (default: 4)
  --act str             Activation function in transformer feed-forward, either relu or gelu.
                        (default: relu)
  --mode {2D,axial,intercell,intracell,none}
                        Attention mode. (default: axial)
  --transf_hsz int      Hidden dimension size in the transformer. (default: 64)
  --n_heads int         Number of self-attention heads. (default: 8)
  --head_dim int        Hidden dimensionality of each head. (default: 8)
  --window int          Window size of row-wise sliding window attention, should be odd. (default: 21)
  --layernorm boolean   Whether to apply layernorm in transformer modules. (default: True)
  --CNN_do float        Dropout rate in the CNN to embed DNA context. (default: 0.0)
  --transf_do float     Dropout rate on the self-attention matrix. (default: 0.2)
  --lr float            Learning rate. (default: 0.0005)
  --lr_decay_factor float
                        Learning rate multiplicative decay applied after every epoch. (default:
                        0.9)
  --warmup_steps int    Number of steps over which the learning rate will linearly warm up.
                        (default: 1000)
```
    
</details>

<details><summary>Logging arguments</summary>
    
```
Logging:
  Logging arguments

  --tensorboard boolean
                        Whether to use tensorboard. If True, then training progress can be
                        followed by using (1) `tensorboard --logdir logfolder/` in a separate
                        terminal and (2) accessing at localhost:6006. (default: True)
  --log_folder str      Folder where the tensorboard logs will be saved. Will additinally contain
                        saved model checkpoints. (default: logfolder)
  --experiment_name str
                        Name of the run within the log folder. (default: experiment)
  --earlystop boolean   Whether to use early stopping after the validation loss has not decreased
                        for `patience` epochs. (default: True)
  --patience int        Number of epochs to wait for a possible decrease in validation loss
                        before early stopping. (default: 10)
```
    
</details>

<details><summary>PyTorch Lightning Trainer arguments</summary>
    
```
pl.Trainer:
  --logger [str_to_bool]
                        Logger (or iterable collection of loggers) for experiment tracking. A
                        ``True`` value uses the default ``TensorBoardLogger``. ``False`` will
                        disable logging. (default: True)
  --checkpoint_callback [str_to_bool]
                        If ``True``, enable checkpointing. It will configure a default
                        ModelCheckpoint callback if there is no user-defined ModelCheckpoint in
                        :paramref:`~pytorch_lightning.trainer.trainer.Trainer.callbacks`.
                        (default: True)
  --default_root_dir str
                        Default path for logs and weights when no logger/ckpt_callback passed.
                        Default: ``os.getcwd()``. Can be remote file paths such as
                        `s3://mybucket/path` or 'hdfs://path/' (default: None)
  --gradient_clip_val float
                        0 means don't clip. (default: 0.0)
  --gradient_clip_algorithm str
                        'value' means clip_by_value, 'norm' means clip_by_norm. Default: 'norm'
                        (default: norm)
  --process_position int
                        orders the progress bar when running multiple models on same machine.
                        (default: 0)
  --num_nodes int       number of GPU nodes for distributed training. (default: 1)
  --num_processes int   number of processes for distributed training with
                        distributed_backend="ddp_cpu" (default: 1)
  --gpus _gpus_allowed_type
                        number of gpus to train on (int) or which GPUs to train on (list or str)
                        applied per node (default: None)
  --auto_select_gpus [str_to_bool]
                        If enabled and `gpus` is an integer, pick available gpus automatically.
                        This is especially useful when GPUs are configured to be in "exclusive
                        mode", such that only one process at a time can access them. (default:
                        False)
  --tpu_cores _gpus_allowed_type
                        How many TPU cores to train on (1 or 8) / Single TPU to train on [1]
                        (default: None)
  --log_gpu_memory str  None, 'min_max', 'all'. Might slow performance (default: None)
  --progress_bar_refresh_rate int
                        How often to refresh progress bar (in steps). Value ``0`` disables
                        progress bar. Ignored when a custom progress bar is passed to
                        :paramref:`~Trainer.callbacks`. Default: None, means a suitable value
                        will be chosen based on the environment (terminal, Google COLAB, etc.).
                        (default: None)
  --overfit_batches _int_or_float_type
                        Overfit a fraction of training data (float) or a set number of batches
                        (int). (default: 0.0)
  --track_grad_norm float
                        -1 no tracking. Otherwise tracks that p-norm. May be set to 'inf'
                        infinity-norm. (default: -1)
  --check_val_every_n_epoch int
                        Check val every n train epochs. (default: 1)
  --fast_dev_run [str_to_bool_or_int]
                        runs n if set to ``n`` (int) else 1 if set to ``True`` batch(es) of
                        train, val and test to find any bugs (ie: a sort of unit test). (default:
                        False)
  --accumulate_grad_batches int
                        Accumulates grads every k batches or as set up in the dict. (default: 1)
  --max_epochs int      Stop training once this number of epochs is reached. Disabled by default
                        (None). If both max_epochs and max_steps are not specified, defaults to
                        ``max_epochs`` = 1000. (default: None)
  --min_epochs int      Force training for at least these many epochs. Disabled by default
                        (None). If both min_epochs and min_steps are not specified, defaults to
                        ``min_epochs`` = 1. (default: None)
  --max_steps int       Stop training after this number of steps. Disabled by default (None).
                        (default: None)
  --min_steps int       Force training for at least these number of steps. Disabled by default
                        (None). (default: None)
  --max_time str        Stop training after this amount of time has passed. Disabled by default
                        (None). The time duration can be specified in the format DD:HH:MM:SS
                        (days, hours, minutes seconds), as a :class:`datetime.timedelta`, or a
                        dictionary with keys that will be passed to :class:`datetime.timedelta`.
                        (default: None)
  --limit_train_batches _int_or_float_type
                        How much of training dataset to check (float = fraction, int =
                        num_batches) (default: 1.0)
  --limit_val_batches _int_or_float_type
                        How much of validation dataset to check (float = fraction, int =
                        num_batches) (default: 1.0)
  --limit_test_batches _int_or_float_type
                        How much of test dataset to check (float = fraction, int = num_batches)
                        (default: 1.0)
  --limit_predict_batches _int_or_float_type
                        How much of prediction dataset to check (float = fraction, int =
                        num_batches) (default: 1.0)
  --val_check_interval _int_or_float_type
                        How often to check the validation set. Use float to check within a
                        training epoch, use int to check every n steps (batches). (default: 1.0)
  --flush_logs_every_n_steps int
                        How often to flush logs to disk (defaults to every 100 steps). (default:
                        100)
  --log_every_n_steps int
                        How often to log within steps (defaults to every 50 steps). (default: 50)
  --accelerator str     Previously known as distributed_backend (dp, ddp, ddp2, etc...). Can also
                        take in an accelerator object for custom hardware. (default: None)
  --sync_batchnorm [str_to_bool]
                        Synchronize batch norm layers between process groups/whole world.
                        (default: False)
  --precision int       Double precision (64), full precision (32) or half precision (16). Can be
                        used on CPU, GPU or TPUs. (default: 32)
  --weights_summary str
                        Prints a summary of the weights when training begins. (default: top)
  --weights_save_path str
                        Where to save weights if specified. Will override default_root_dir for
                        checkpoints only. Use this if for whatever reason you need the
                        checkpoints stored in a different place than the logs written in
                        `default_root_dir`. Can be remote file paths such as `s3://mybucket/path`
                        or 'hdfs://path/' Defaults to `default_root_dir`. (default: None)
  --num_sanity_val_steps int
                        Sanity check runs n validation batches before starting the training
                        routine. Set it to `-1` to run all batches in all validation dataloaders.
                        (default: 2)
  --truncated_bptt_steps int
                        Deprecated in v1.3 to be removed in 1.5. Please use :paramref:`~pytorch_l
                        ightning.core.lightning.LightningModule.truncated_bptt_steps` instead.
                        (default: None)
  --resume_from_checkpoint str
                        Path/URL of the checkpoint from which training is resumed. If there is no
                        checkpoint file at the path, start from scratch. If resuming from mid-
                        epoch checkpoint, training will start from the beginning of the next
                        epoch. (default: None)
  --profiler str        To profile individual steps during training and assist in identifying
                        bottlenecks. (default: None)
  --benchmark [str_to_bool]
                        If true enables cudnn.benchmark. (default: False)
  --deterministic [str_to_bool]
                        If true enables cudnn.deterministic. (default: False)
  --reload_dataloaders_every_epoch [str_to_bool]
                        Set to True to reload dataloaders every epoch. (default: False)
  --auto_lr_find [str_to_bool_or_str]
                        If set to True, will make trainer.tune() run a learning rate finder,
                        trying to optimize initial learning for faster convergence.
                        trainer.tune() method will set the suggested learning rate in self.lr or
                        self.learning_rate in the LightningModule. To use a different key set a
                        string instead of True with the key name. (default: False)
  --replace_sampler_ddp [str_to_bool]
                        Explicitly enables or disables sampler replacement. If not specified this
                        will toggled automatically when DDP is used. By default it will add
                        ``shuffle=True`` for train sampler and ``shuffle=False`` for val/test
                        sampler. If you want to customize it, you can set
                        ``replace_sampler_ddp=False`` and add your own distributed sampler.
                        (default: True)
  --terminate_on_nan [str_to_bool]
                        If set to True, will terminate training (by raising a `ValueError`) at
                        the end of each training batch, if any of the parameters or the loss are
                        NaN or +/-inf. (default: False)
  --auto_scale_batch_size [str_to_bool_or_str]
                        If set to True, will `initially` run a batch size finder trying to find
                        the largest batch size that fits into memory. The result will be stored
                        in self.batch_size in the LightningModule. Additionally, can be set to
                        either `power` that estimates the batch size through a power search or
                        `binsearch` that estimates the batch size through a binary search.
                        (default: False)
  --prepare_data_per_node [str_to_bool]
                        If True, each LOCAL_RANK=0 will call prepare data. Otherwise only
                        NODE_RANK=0, LOCAL_RANK=0 will prepare data (default: True)
  --plugins str         Plugins allow modification of core behavior like ddp and amp, and enable
                        custom lightning plugins. (default: None)
  --amp_backend str     The mixed precision backend to use ("native" or "apex") (default: native)
  --amp_level str       The optimization level to use (O1, O2, etc...). (default: O2)
  --distributed_backend str
                        deprecated. Please use 'accelerator' (default: None)
  --move_metrics_to_cpu [str_to_bool]
                        Whether to force internal logged metrics to be moved to cpu. This can
                        save some gpu memory, but can make training slower. Use with attention.
                        (default: False)
  --multiple_trainloader_mode str
                        How to loop over the datasets when there are multiple train loaders. In
                        'max_size_cycle' mode, the trainer ends one epoch when the largest
                        dataset is traversed, and smaller datasets reload when running out of
                        their data. In 'min_size' mode, all the datasets reload when reaching the
                        minimum length of datasets. (default: max_size_cycle)
  --stochastic_weight_avg [str_to_bool]
                        Whether to use `Stochastic Weight Averaging (SWA)
                        <https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-
                        averaging/>_` (default: False)
```
    
</details>

### Imputation and denoising <a name="impute"></a>

Once a model is trained on already-observed sites, it can be used to impute unobserved sites. We provide `impute_genome.py` for genome-wide imputation using either CpG Transformer, DeepCpG or CaMelia models. For all methods, the same three inputs are expected as with the training scripts: `X.npz`, `y.npz` and `pos.npz`. In addition, an output location e.g. `output.npz` should be specified. By default, `impute_genome.py` will return model predictions for every chromosome. To change this behavior, chromosomes can also be selected based on the `--keys` flag. The script will also predict every site irregardless of whether they are observer or not. To modulate this behavior, the `--denoise` flag can be set to `False`. Doing this will make it so that the output file will only impute unobserved methylation states and not impute/denoise observed methylation states. It has to be noted that CpG Transformer is the only model to explicitly model denoising in its design (objective function). For more detailed information about additional flags, use `python impute_genome.py -h`.

In terms of speed, CpG Transformer and DeepCpG will be fastest (when using a GPU) and CaMelia slowest due to its heavy preprocessing. (I have tried my best to vectorize and parallellize as much as possible, any contributors interested in improving are welcome to open a pull request.)


### Benchmarking <a name="benchmark"></a>

Because CpG Transformer masks and randomizes multiple CpG sites per batch, performances reported during training are negatively biased. In practical use (imputation), this forms no problem, as no masking takes place at this point. For benchmarking however, CaMelia and DeepCpG may have an unfair advantage because they do not train using masking strategies. To perform a fair comparison between all models, a separate benchmarking script `benchmark.py` is provided for CpG Transformer. To benchmark DeepCpG and CaMelia, `impute_genome.py` script can be used with one or more test chromosomes specified. It has to be noted that CpG Transformer was not designed to separately mask and predict sites like this. Consequently, the benchmark script may take a lot of time to run.


### Interpretation <a name="interpret"></a>

See the `./interpretation` folder.

## Pre-trained models <a name="pretrained"></a>

Pre-trained CpG Transformer models for all tested [datasets](#perf-comp) are available as PyTorch model state dicts in `data/model_checkpoints/`.

## Citation <a name="citation"></a>

If you find this repository useful in your research, please cite our [paper](https://www.biorxiv.org/content/10.1101/2021.06.08.447547v2).
```bibtex
@article {dewaele2021cpg,
	author = {De Waele, Gaetan and Clauwaert, Jim and Menschaert, Gerben and Waegeman, Willem},
	title = {CpG Transformer for imputation of single-cell methylomes},
    journal = {Bioinformatics},
    year = {2021},
    month = {10},
	issn = {1367-4803},
    doi = {10.1093/bioinformatics/btab746},
    url = {https://doi.org/10.1093/bioinformatics/btab746},
    note = {btab746},
}
```


## License <a name="license"></a>

This source code is licensed under the MIT license found in the `LICENSE` file in the root directory of this source tree.
