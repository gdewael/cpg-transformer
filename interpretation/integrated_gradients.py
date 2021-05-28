import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
import argparse
import sys
sys.path.append('..')
import pandas as pd
from cpg_transformer.cpgtransformer import CpGTransformer

def boolean(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

        
class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.MetavarTypeHelpFormatter):
    pass
parser = ArgumentParser(description='Interpretation script. Only works with GPU.' ,
                        formatter_class=CustomFormatter)

parser.add_argument('X', type=str, metavar='X', help='NumPy file (.npz) containing encoded genome.')
parser.add_argument('y', type=str, metavar='y', help='NumPy file (.npz) containing input methylation matrix.')
parser.add_argument('pos', type=str, metavar='pos', help='NumPy file (.npz) containing positions of CpG sites.')
parser.add_argument('output', type=str, metavar='output', help='NumPy file (.npz) containing output contributions.')

parser.add_argument('--model_checkpoint', type=str, default=None,
                    help='.ckpt file containing the model to use. DOES NOT WORK WITH .pt STATE DICT FILES.')

parser.add_argument('--config_file', type=str, default=None,
                    help='config file specifying which sites to interpret and how. See README for more info.')

parser.add_argument('--make_plot', type=boolean, default=False,
                    help='Whether to make plots. See README for example plot')
parser.add_argument('--plot_name_prefix', type=str, default='interpret_model_',
                    help='If making plots, which prefix for the filenames.')
parser.add_argument('--plot_name_suffix', type=str, default='.pdf',
                    help='If making plots, which suffix for the filename (file extension)')

parser.add_argument('--which', type=str, choices=['main', 'embeds', 'both'], default='both',
                    help='What to interpret. If `main`, will interpret all embeddings combined as input to the transformer layers. If `embeds`, will interpret (and plot) each embedding separately. `both` naturally does both')

parser.add_argument('--figsize', type=int, nargs='+', default=[10,3],
                    help='2 integers specifying the figure size.')


args = parser.parse_args()

print('----- Integrated Gradients interpretation -----')

X = np.load(args.X)
y = np.load(args.y)
pos = np.load(args.pos)

model = CpGTransformer.load_from_checkpoint(args.model_checkpoint)
model = model.to('cuda')
model.eval()

RF_TF = (model.hparams.window-1)*model.hparams.n_transformers+1
RF2_TF = int((RF_TF-1)/2)
r = torch.arange(-int((model.RF-1)/2),int((model.RF-1)/2)+1)

config = pd.read_csv(args.config_file,header=None)

assert set([0,1]).union(set(np.unique(config[3].values))) == set([0,1]), '4th column of config file should contain only 0s or 1s'


class InterpreterCaptum(nn.Module):
    def __init__(self, model):
        super().__init__()
        
        self.model = model
        self.tfw =  int((model.hparams['window']-1)/2)
        
        self.cell_embed_weight = self.model.cell_embed.weight.data
        self.CpG_embed_weight = self.model.CpG_embed.weight.data
        self.CNN_embed_weight = self.model.CNN[0].embed.weight.data
        
    def forward(self, CNN_embed, CpG_embed, cell_embed, pos, index_label, y_true):
        
        x = torch.cat((CpG_embed, cell_embed, CNN_embed), -1)
        x = self.model.combine_embeds(x)

        for layer in self.model.transformer:
            x, pos = layer((x,pos))
            x = x[:,self.tfw:-self.tfw]
            pos = pos[:,self.tfw:-self.tfw]
        y_predict = self.model.output_head(x).squeeze(1).squeeze(-1)
        y_hat = y_predict[0, index_label]
        return torch.abs(torch.sigmoid(y_hat)-torch.abs(y_true-1))

    
def plotter(to_plot, annot, y_true, index_label, pred, title, vlim, figsize):
    
    plt.figure(figsize=figsize)
    ax = sns.heatmap(to_plot, annot=annot, fmt = '', cbar_kws={'aspect':10}, annot_kws={'size':8},
                     cmap=sns.diverging_palette(217, 134, s=85,l=50, as_cmap=True),
                     vmin=-vlim,vmax=vlim)
    ax.set_title('{}\nReference label: {}, predicted for cell: {}, Predicted probability of reference class: {:.4f}'.format(title, y_true[0].cpu().item(), index_label,
                 pred.item()))
    tl = ax.get_yticklabels()
    _ = ax.set_yticklabels(tl, rotation=0)
    ax.set_xlabel('CpG site relative from prediction site')
    ax.set_ylabel('Cell index')
    ax.set_xticks(np.arange(0,81,10)+0.5)
    _ = ax.set_xticklabels(np.arange(-40,41,10))
    plt.tight_layout()
    
    
InterpreterNet = InterpreterCaptum(model)
ig = IntegratedGradients(InterpreterNet)

cell_embed_weight = model.cell_embed.weight.data
CpG_embed_weight = model.CpG_embed.weight.data
CNN_embed_weight = model.CNN[0].embed.weight.data


out_dict = {'preds': [], 'ref': []}
if args.which == 'main':
    out_dict['total'] = []
if args.which == 'embeds':
    out_dict['cell'] = []
    out_dict['dna'] = []
    out_dict['cpg'] = []
if args.which == 'both':
    out_dict['cell'] = []
    out_dict['dna'] = []
    out_dict['cpg'] = []
    out_dict['total'] = []
    

for i in range(config.shape[0]):
    print('Interpreting row number', i, "...", end='\r')
    row = config.iloc[i,:]
    key_s = row[0]
    row_index_s = row[1]
    col_index_s = row[2]
    ref_label_s = None if row[3] == 'None' else int(row[3])
    label_changes_s = None if row[4] == 'None' else int(row[4])
    
    y_batch = torch.from_numpy(y[key_s][max(0,col_index_s-RF2_TF):col_index_s+RF2_TF+1]).to('cuda')
    pos_batch = torch.from_numpy(pos[key_s][max(0,col_index_s-RF2_TF):col_index_s+RF2_TF+1])
    X_batch = torch.from_numpy(X[key_s][pos_batch.unsqueeze(1).repeat(1,model.RF)+r]).to('cuda')
    pos_batch = pos_batch.to('cuda').unsqueeze(0)
    cell_indices = torch.arange(y_batch.shape[1]).to('cuda')

    y_true = y[key_s][col_index_s, row_index_s] if ref_label_s is None else ref_label_s
    y_true = torch.tensor(y_true).unsqueeze(0).to('cuda')
    if label_changes_s is not None:
        y_batch[col_index_s-max(0,col_index_s-RF2_TF), row_index_s] = label_changes_s
    y_batch += 1
    
    with torch.no_grad():
        y_batch = y_batch.to(torch.long).unsqueeze(0)
        X_batch = X_batch.to(torch.long).unsqueeze(0)
        cell_indices = cell_indices.to(torch.long).unsqueeze(0)

        bsz, seqlen, n_cells = y_batch.shape[:3]

        pos_batch = (pos_batch - pos_batch[:,0].unsqueeze(1)).to(torch.float)

        cell_embed = F.one_hot(cell_indices, num_classes=cell_embed_weight.shape[0]).to(torch.float)
        CpG_embed = F.one_hot(y_batch,num_classes=CpG_embed_weight.shape[0]).to(torch.float)
        CNN_embed = F.one_hot(X_batch, num_classes=CNN_embed_weight.shape[0]).to(torch.float)

        cell_embed = torch.matmul(cell_embed, cell_embed_weight)
        CpG_embed = torch.matmul(CpG_embed, CpG_embed_weight)
        CNN_embed = torch.matmul(CNN_embed, CNN_embed_weight)

        CNN_embed = model.CNN[0].CNN(CNN_embed.view(-1,model.RF,4).permute(0,2,1)).view(-1,256*model.CNN[0].hlen)
        CNN_embed = model.CNN[0].lin(CNN_embed)
        DNA_embed = model.CNN[1:](CNN_embed).view(bsz, seqlen, -1)


        DNA_embed = DNA_embed.unsqueeze(-2).expand(-1,-1,n_cells,-1)
        cell_embed = cell_embed.unsqueeze(1).expand(-1,seqlen,-1,-1)

    out = ig.attribute(inputs=(DNA_embed, CpG_embed, cell_embed), additional_forward_args=(pos_batch, row_index_s, y_true),
                   internal_batch_size=1)
    
    for param in model.parameters():
        param.grad = None
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        pred = InterpreterNet(DNA_embed, CpG_embed, cell_embed, pos_batch, row_index_s, y_true)


    out = [o.detach().cpu() for o in out]

    
    out_dict['ref'].append(pred[0].cpu().item())
    out_dict['preds'].append(y_true[0].cpu().item())
    if args.which == 'both':
        out_dict['total'].append(torch.sum(out[-1][0]+out[0][0]+out[1][0],-1).cpu())
        out_dict['dna'].append(torch.sum(out[0][0],-1).cpu())
        out_dict['cpg'].append(torch.sum(out[1][0],-1).cpu())
        out_dict['cell'].append(torch.sum(out[2][0],-1).cpu())
    elif args.which == 'embeds':
        out_dict['dna'].append(torch.sum(out[0][0],-1).cpu())
        out_dict['cpg'].append(torch.sum(out[1][0],-1).cpu())
        out_dict['cell'].append(torch.sum(out[2][0],-1).cpu())
    elif args.which == 'main':
        out_dict['total'].append(torch.sum(out[-1][0]+out[0][0]+out[1][0],-1).cpu())
    
    if args.make_plot:
        annot = (y_batch[0].cpu().numpy().T-1).astype(str)
        annot[np.where(annot=="-1")] = '?'

        if args.which != 'embeds':
            to_plot = torch.sum(out[-1][0]+out[0][0]+out[1][0],-1).cpu().numpy().T
            vlim = np.abs(to_plot).max()
            plotter(to_plot, annot, y_true, row_index_s, pred, 'Total Contributions', vlim, args.figsize)
            plt.savefig(args.plot_name_prefix+'total_'+f"{i:03d}"+args.plot_name_suffix)

        if args.which != 'main':
            to_plot = torch.sum(out[1][0],dim=-1).cpu().numpy().T
            vlim = np.abs(to_plot).max()
            plotter(to_plot, annot, y_true, row_index_s, pred, 'CpG Embedding Contributions', vlim, args.figsize)
            plt.savefig(args.plot_name_prefix+'cpg_'+f"{i:03d}"+args.plot_name_suffix)

            to_plot = torch.sum(out[-1][0],dim=-1).cpu().numpy().T
            vlim = np.abs(to_plot).max()
            plotter(to_plot, annot, y_true, row_index_s, pred, 'Cell Embedding Contributions', vlim, args.figsize)
            plt.savefig(args.plot_name_prefix+'cell_'+f"{i:03d}"+args.plot_name_suffix)

            to_plot = torch.sum(out[0][0],dim=-1).cpu().numpy().T
            vlim = np.abs(to_plot).max()
            plotter(to_plot, annot, y_true, row_index_s, pred, 'DNA Embedding Contributions', vlim, args.figsize)
            plt.savefig(args.plot_name_prefix+'dna_'+f"{i:03d}"+args.plot_name_suffix)

        plt.close('all')
    
out_dict['preds'] = np.array(out_dict['preds'])
out_dict['ref'] = np.array(out_dict['ref'])
if args.which == 'both':
    out_dict['total'] = np.array([p.numpy() for p in out_dict['total']])
    out_dict['cpg'] = np.array([p.numpy() for p in out_dict['cpg']])
    out_dict['dna'] = np.array([p.numpy() for p in out_dict['dna']])
    out_dict['cell'] = np.array([p.numpy() for p in out_dict['cell']])
elif args.which == 'embeds':
    out_dict['cpg'] = np.array([p.numpy() for p in out_dict['cpg']])
    out_dict['dna'] = np.array([p.numpy() for p in out_dict['dna']])
    out_dict['cell'] = np.array([p.numpy() for p in out_dict['cell']])
elif args.which == 'main':
    out_dict['total'] = np.array([p.numpy() for p in out_dict['total']])
    
np.savez_compressed(args.output, **out_dict)  

print()
print('Done')