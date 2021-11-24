import numpy as np
import torch
from argparse import ArgumentParser
import argparse

def boolean(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

        
class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.MetavarTypeHelpFormatter):
    pass
parser = ArgumentParser(description='Genome-wide imputation script. Outputs a methylation matrix in the same format as the input `y.npz`, where every element is a floating number between 0 and 1 representing the model prediction.' ,
                        formatter_class=CustomFormatter)

parser.add_argument('model', type=str, choices=['cpg_transformer', 'deepcpg', 'camelia'],
                    help='Which model type to use for imputation.')

parser.add_argument('X', type=str, metavar='X', help='NumPy file (.npz) containing encoded genome.')
parser.add_argument('y', type=str, metavar='y', help='NumPy file (.npz) containing input methylation matrix.')
parser.add_argument('pos', type=str, metavar='pos', help='NumPy file (.npz) containing positions of CpG sites.')
parser.add_argument('output', type=str, metavar='output', help='NumPy file (.npz) containing output methylation matrix.')

optional_parse = parser.add_argument_group('General optional arguments.')

optional_parse.add_argument('--keys', type=str, nargs='+', default=None,
                            help='Only impute chromosomes corresponding to these keys.')
optional_parse.add_argument('--denoise', type=boolean, default=True,
                            help='If False, return the original methylation state for already-observed elements in the output. In other words: only unobserved elements will be imputed and observed sites will retain their original label always. If True, model predictions will be returned for all inputs, irregardless of whether they are observed.')

cpgtf_parse = parser.add_argument_group('CpG Transformer-specific arguments.',
                                        'These arguments are only relevant when imputing with CpG Transformer models.')
cpgtf_parse.add_argument('--model_checkpoint', type=str, default=None,
                         help='.ckpt file containing the model to use. DOES NOT WORK WITH .pt STATE DICT FILES.')
cpgtf_parse.add_argument('--segment_size', type=int, default=1024,
                         help='Bin size in number of CpG sites (columns) that every batch will contain.')
cpgtf_parse.add_argument('--n_workers', type=int, default=4,
                      help='Number of worker threads to use in data loading. Increase if you experience a CPU bottleneck.')
cpgtf_parse.add_argument('--device', type=str, choices=['CPU', 'GPU'], default='GPU',
                         help='GPU or CPU. For inference, it is currently only possible to use 1 GPU.')


deepcpg_parse = parser.add_argument_group('DeepCpG-specific arguments.',
                                        'These arguments are only relevant when imputing with DeepCpG models.')
deepcpg_parse.add_argument('--batch_size_dpcpg', type=int, default=512,
                         help='Number of CpG sites to impute in parallel.')
deepcpg_parse.add_argument('--model_checkpoint_dpcpg', type=str, default=None,
                         help='.ckpt file containing the model to use.')
deepcpg_parse.add_argument('--n_workers_dpcpg', type=int, default=4,
                      help='Number of worker threads to use in data loading. Increase if you experience a CPU bottleneck.')
deepcpg_parse.add_argument('--device_dpcpg', type=str, choices=['CPU', 'GPU'], default='GPU',
                         help='GPU or CPU. For inference, it is currently only possible to use 1 GPU.')
deepcpg_parse.add_argument('--window_dpcpg', type=int, default=25,
                      help='How many sites to use up and downstream to make up local context in RNN. The same value used in training should be used.')
deepcpg_parse.add_argument('--maxdist_dpcpg', type=int, default=25000,
                         help='Maximal distance over which relative distances are normalized. CpG sites farther away will get assigned the max relative distance (1). The same value used in training should be used.')

camelia_parse = parser.add_argument_group('CaMelia-specific arguments.',
                                        'These arguments are only relevant when imputing with CaMelia models.')
camelia_parse.add_argument('--camelia_folder', type=str, default=None,
                         help='Folder containing the cbm files. Model filenames should end in the cell index number with 3 leading zeros + .cbm (e.g. ...001.cbm and ...015.cbm)')
camelia_parse.add_argument('--neigh', type=boolean, default=True,
                      help='Whether your models were trained with neighboring methylation features.')
camelia_parse.add_argument('--local', type=boolean, default=True,
                      help='Whether your models were trained with locally paired similarity feature.')
camelia_parse.add_argument('--threshold', type=float, default=0.8,
                      help='Threshold on locally paired similarity feature used in training of the models.')
camelia_parse.add_argument('--DNA', type=boolean, default=True,
                      help='Whether your models were trained with DNA context features.')
camelia_parse.add_argument('--no_nan_pred', type=boolean, default=False,
                      help='Corresponds to the --dropnans argument in `train_camelia.py`. script. If True, will not predict sites that have an undefined locally paired similarity feature. In the final imputed matrix, these sites will have an `-1` output.')
camelia_parse.add_argument('--batch_size_camelia', type=int, default=50000,
                         help='Number of CpG sites per cell to impute in parallel.')

args = parser.parse_args()


if args.model == 'cpg_transformer':
    print('----- Imputing with CpG Transformer -----')
    from cpg_transformer.cpgtransformer import CpGTransformer
    from cpg_transformer.datamodules import CpGTransformerImputingDataModule
    
    X = np.load(args.X)
    y = np.load(args.y)
    pos = np.load(args.pos)

    dev = 'cuda' if args.device == 'GPU' else 'cpu'
    
    model = CpGTransformer.load_from_checkpoint(args.model_checkpoint)
    model.eval()
    model = model.to(dev)

    RF_TF = (model.hparams.window-1)*model.hparams.n_transformers+1
    RF2_TF = int((RF_TF-1)/2)
    dm = CpGTransformerImputingDataModule(X, y, pos, segment_size=args.segment_size, RF=model.hparams.RF, RF_TF=RF_TF,
                                     keys=args.keys, n_workers=args.n_workers)
    print('Preprocessing data ...')
    dm.setup(None)

    y_outputs = dict()
    for key, loader in dm.datasets_per_chr.items():
        print('Imputing', key, '...')
        y_outputs_key = np.empty(y[key].shape)

        lenloader_key = len(loader)
        for ix, batch in enumerate(loader):
            
            batch = [batch[0].to(dev, torch.long), batch[1].to(dev, model.dtype),
                     batch[2].to(dev, torch.long), batch[3].to(dev, torch.long)]
            with torch.no_grad():
                output = model(*batch)
            
            if model.hparams.data_mode == 'binary':
                output = torch.sigmoid(output[0]).to('cpu').numpy()
            elif model.hparams.data_mode == 'continuous':
                output = output[0].to('cpu').numpy()
                
            if ix == 0:
                y_outputs_key[:args.segment_size-RF2_TF] = output[:args.segment_size-RF2_TF]
            elif ix+1 == lenloader_key:
                loc = (args.segment_size-RF2_TF*2)*ix
                y_outputs_key[loc+RF2_TF:] = output[RF2_TF:]
            else:
                loc = (args.segment_size-RF2_TF*2)*ix
                y_outputs_key[loc+RF2_TF:loc+args.segment_size-RF2_TF] = output[RF2_TF:-RF2_TF]

        y_outputs[key] = y_outputs_key

    if args.denoise == False:
        for key in y_outputs.keys():
            observed = np.where(y[key] != -1)
            y_outputs[key][observed] = y[key][observed]
            
    np.savez_compressed(args.output, **y_outputs)
    

elif args.model == 'deepcpg':
    print('----- Imputing with DeepCpG -----')

    X = np.load(args.X)
    y = np.load(args.y)
    pos = np.load(args.pos)
    
    if args.keys is not None:
        X = {k:X[k] for k in keys}
        y = {k:y[k] for k in keys}
        pos = {k:pos[k] for k in keys}

    dev = 'cuda' if args.device_dpcpg == 'GPU' else 'cpu'
    from cpg_transformer.deepcpg import DeepCpG
    from cpg_transformer.datamodules import DeepCpGDataModule

    model = DeepCpG.load_from_checkpoint(args.model_checkpoint_dpcpg)
    model.eval()
    model = model.to(dev)
    dm = DeepCpGDataModule(X, y, pos, model.RF, batch_size=args.batch_size_dpcpg,
                           n_workers=args.n_workers_dpcpg, window=args.window_dpcpg,
                           max_dist=args.maxdist_dpcpg, batch_index=True)

    print('Preprocessing data ...')
    dm.setup(None)
    loader = torch.utils.data.DataLoader(dm.train, num_workers=dm.nw, 
                                batch_size=dm.bsz, shuffle=False, pin_memory=True)

    y_outputs = {k:np.empty(y[k].shape) for k in y.keys()}

    names_chr = list(y.keys())

    c = 0
    print('Imputing', names_chr[c], '...')
    for batch in loader:
        DNA, CpG, _, pos = batch
        DNA, CpG = DNA.to(dev, dtype=torch.long), CpG.to(dev, dtype=model.dtype)
        with torch.no_grad():
            y_out = model(DNA, CpG)
        y_out = torch.sigmoid(y_out).cpu().numpy()

        if len(np.unique(pos[0])) == 1:
            y_outputs[names_chr[pos[0][0]]][pos[1]] = y_out
        else:
            c+=1; print('Imputing', names_chr[c], '...')
            for ch in np.unique(pos[0]):
                ix_ = (pos[0]==ch)
                y_outputs[names_chr[pos[0][ix_][0]]][pos[1][ix_]] = y_out[ix_]
                
                
    if args.denoise == False:
        for key in y_outputs.keys():
            observed = np.where(y[key] != -1)
            y_outputs[key][observed] = y[key][observed]
            
    np.savez_compressed(args.output, **y_outputs)
                
elif args.model == 'camelia':
    print('----- Imputing with CaMelia -----')

    X = np.load(args.X)
    y = np.load(args.y)
    pos = np.load(args.pos)

    if keys is not None:
        X = {k:X[k] for k in keys}
        y = {k:y[k] for k in keys}
        pos = {k:pos[k] for k in keys}

    from cpg_transformer.datamodules import CaMeliaPreprocessor
    from catboost import CatBoostClassifier    
    import pandas as pd

    y_outputs = {k:np.full(y[k].shape, -1) for k in y.keys()}

    n_cells = y[list(y.keys())[0]].shape[1]
    for cell in range(n_cells):
        print('----- Cell', cell, '-----')

        model_cell = [f for f in os.listdir(args.camelia_folder) if f"{cell:03d}.cbm" in f][0]
        model = CatBoostClassifier().load_model(args.camelia_folder.rstrip('/')+'/'+model_cell)
        print('Preprocessing ...')
        prepr = CaMeliaPreprocessor(X,y,pos, val_keys=None, test_keys=None, fracs=[1,0,0])

        prepr_out = prepr(cell, neigh=args.neigh, local=args.local, DNA=args.DNA,
                          threshold=args.threshold, whole_genome=True)

        X_in = prepr_out[0]

        chroms = [0]+list(np.cumsum([y_.shape[0] for y_ in y.values()]))

        X_chrs = [X_in.iloc[chroms[k]:chroms[k+1]] for k in range(len(chroms)-1)]

        for chr_name, X_chr in zip(y.keys(), X_chrs):
            print('Imputing', chr_name, '...')
            for i in range(0,X_chr.shape[0],args.batch_size_camelia):
                X_slice = X_chr.iloc[i:i+args.batch_size_camelia]
                if args.no_nan_pred:
                    drop = pd.isnull(X_slice).values.sum(-1) == 0
                    y_outputs[chr_name][i:i+batch_size, cell][drop] = model.predict_proba(X_slice[drop])[:,1]
                else:
                    y_outputs[chr_name][i:i+batch_size, cell] = model.predict_proba(X_slice)[:,1]

    if args.denoise == False:
        for key in y_outputs.keys():
            observed = np.where(y[key] != -1)
            y_outputs[key][observed] = y[key][observed]
                
    np.savez_compressed(args.output, **y_outputs)