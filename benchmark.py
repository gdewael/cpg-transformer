import numpy as np
import torch
from argparse import ArgumentParser
import argparse

        
class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.MetavarTypeHelpFormatter):
    pass
parser = ArgumentParser(description='Benchmarking script for CpG Transformer. Masks and predicts every site separately.' ,
                        formatter_class=CustomFormatter)

parser.add_argument('X', type=str, metavar='X', help='NumPy file (.npz) containing encoded genome.')
parser.add_argument('y', type=str, metavar='y', help='NumPy file (.npz) containing input methylation matrix.')
parser.add_argument('pos', type=str, metavar='pos', help='NumPy file (.npz) containing positions of CpG sites.')
parser.add_argument('output', type=str, metavar='output', help='NumPy file (.npz) containing output methylation matrix.')

parser.add_argument('--key', type=str, default='chr10',
                    help='Only impute the chromosome corresponding to this key.')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size. Larger batch sizes that just about fit in memory will result in optimal benchmarking speed.')
parser.add_argument('--device', type=str, choices=['CPU', 'GPU'], default='GPU',
                         help='GPU or CPU. For inference, it is currently only possible to use 1 GPU.')
parser.add_argument('--model_checkpoint', type=str, default=None,
                    help='.ckpt file containing the model to use.')
parser.add_argument('--n_workers', type=int, default=4,
                    help='Number of worker threads to use in data loading. Increase if you experience a CPU bottleneck.')


args = parser.parse_args()


print('----- Benchmarking CpG Transformer -----')

X = np.load(args.X)
y = np.load(args.y)
pos = np.load(args.pos)


from cpg_transformer.cpgtransformer import CpGTransformer
from cpg_transformer.datamodules import ExhaustiveBenchmarkDNAEmbedding, ExhaustiveBenchmarkDataset

X = X[args.key]
y = y[args.key]
pos = pos[args.key]

dev = 'cuda' if args.device == 'GPU' else 'cpu'


model = CpGTransformer.load_from_checkpoint(args.model_checkpoint)
model.eval()
model = model.to(dev)


RF_TF = (model.hparams.window-1)*model.hparams.n_transformers+1
RF2_TF = int((RF_TF-1)/2)
w = int((model.hparams.window-1)/2)


dataset = ExhaustiveBenchmarkDNAEmbedding(X, pos, RF=model.hparams.RF)
dataloader = torch.utils.data.DataLoader(dataset, num_workers=args.n_workers, batch_size=1024, shuffle=False, pin_memory=True)

with torch.no_grad():
    DNA_windows = [model.CNN(batch.to(dev,torch.long).view(-1,model.RF)).view(batch.shape[0], -1).to('cpu') for batch in dataloader]
DNA_embeddings = torch.cat(DNA_windows)

dataset = ExhaustiveBenchmarkDataset(DNA_embeddings, y, pos, window=RF_TF, RF=model.hparams.RF)
dataloader = torch.utils.data.DataLoader(dataset, num_workers=args.n_workers, batch_size=args.batch_size, shuffle=False, pin_memory=True)
lenloader = len(dataloader)


y_output = np.full(y.shape, -1.)

from time import time
t=time()

with torch.no_grad():
    for ix, batch in enumerate(dataloader):
        DNA, y_input, pos, index_label, cell_indices = batch
        
        DNA_embed = DNA.to(dev)
        y_masked = y_input.to(dev, dtype=torch.long)
        pos = pos.to(dev, dtype=torch.long)
        cells = cell_indices.to(dev, dtype=torch.long)

        bsz, seqlen, n_cells = y_masked.shape[:3]

        pos = pos - pos[:,0].unsqueeze(1)


        cell_embed = model.cell_embed(cells) # n_rep x embed_size
        CpG_embed = model.CpG_embed(y_masked) # bsz, seqlen, n_rep, cpg_size

        DNA_embed = DNA_embed.unsqueeze(-2).expand(-1,-1,n_cells,-1)
        cell_embed = cell_embed.unsqueeze(1).expand(-1,seqlen,-1,-1)
        x = torch.cat((CpG_embed, cell_embed, DNA_embed), -1)
        x = model.combine_embeds(x)
        for layer in model.transformer:
            x, pos = layer((x,pos))
            x = x[:,w:-w]
            pos = pos[:,w:-w]
        y_predict = model.output_head(x).squeeze(1).squeeze(-1)
        y_hat = y_predict[torch.arange(bsz), index_label[1]]

        y_output[index_label[0], index_label[1]] = torch.sigmoid(y_hat).cpu().numpy()


        if (ix % 50) == 0:
            print('Progress:', np.round(ix/lenloader*100,2), '%. After', np.round(time()-t,2), 'seconds', end="\r", flush=True)

    for i in range(dataset.edges.shape[0]):
        b = dataset.edge_getitem(i)    
        DNA, y_input, pos, index_label, cell_indices = b

        DNA, y_input = DNA.unsqueeze(0), y_input.unsqueeze(0)
        pos, cell_indices = pos.unsqueeze(0), cell_indices.unsqueeze(0)

        DNA_embed = DNA.to(dev)
        y_masked = y_input.to(dev, dtype=torch.long)
        pos = pos.to(dev, dtype=torch.long)
        cells = cell_indices.to(dev, dtype=torch.long)

        bsz, seqlen, n_cells = y_masked.shape[:3]

        pos = pos - pos[:,0].unsqueeze(1)


        cell_embed = model.cell_embed(cells) # n_rep x embed_size
        CpG_embed = model.CpG_embed(y_masked) # bsz, seqlen, n_rep, cpg_size

        DNA_embed = DNA_embed.unsqueeze(-2).expand(-1,-1,n_cells,-1)
        cell_embed = cell_embed.unsqueeze(1).expand(-1,seqlen,-1,-1)
        x = torch.cat((CpG_embed, cell_embed, DNA_embed), -1)
        x = model.combine_embeds(x)
        for layer in model.transformer:
            x, pos = layer((x,pos))

        y_predict = model.output_head(x).squeeze(1).squeeze(-1)
        index_in_out = index_label[0]-max(0,index_label[0]-dataset.w)
        y_hat = y_predict[0, index_in_out, index_label[1]]
        y_output[index_label[0], index_label[1]] = torch.sigmoid(y_hat).cpu().numpy()
        
     
    print('Progress:', 100.0, '%. After', np.round(time()-t,2), 'seconds', end="\r", flush=True)   
        
with open(args.output, 'wb') as f:
    np.savez_compressed(f, **{args.key:y_output})