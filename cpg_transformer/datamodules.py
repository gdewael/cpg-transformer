import torch
import torch.nn as nn
import numpy as np
import random
import pytorch_lightning as pl
from torch.utils.data import WeightedRandomSampler
import math
import pandas as pd


def sample_from_cdf(cdf, n):
    return cdf['x'][(torch.rand(n, 1) < cdf['y']).int().argmax(-1)]

class CpGTransformerDataModule(pl.LightningDataModule):
    def __init__(self, X, y, pos, segment_size=1024, RF=1001, fracs=[1,0,0],
                 mask_perc=0.25, mask_random_perc=0.2,
                 resample_cells=None, resample_cells_val=None,
                 val_keys=None, test_keys=None,
                 batch_size=1, n_workers=4):
        
        assert len(fracs)==3,'length of fractions should be 3 for train/val/test'
        assert sum(fracs)==1, 'Sum of train/val/test fractions should be one.'
        assert val_keys is None or type(val_keys) is list, 'val_keys should be None or list'
        assert test_keys is None or type(test_keys) is list, 'test_keys should be None or list'
        if val_keys is not None and test_keys is not None:
            assert set(val_keys) & set(test_keys) == set(), 'No overlap allowed between val_keys & test_keys'
        super().__init__()
        
        self.X = X
        self.y = y
        self.pos = pos
        self.segment_size = segment_size
        self.RF = RF; self.RF2 = int((RF-1)/2)
        self.fracs = fracs
        self.val_keys = val_keys
        self.test_keys = test_keys
        self.mask_perc = mask_perc
        self.mask_random_perc = mask_random_perc
        self.bsz = batch_size
        self.nw = n_workers
        self.resample = resample_cells
        self.resample_val = resample_cells_val
        
    def setup(self, stage):
        train = []; val = []; test = []
        
        for chr_name in self.y.keys():
            y_temp = self.y[chr_name]
            X_temp = self.X[chr_name]
            pos_temp = self.pos[chr_name]
            
            if 'numpy' in str(type(X_temp)):
                X_temp = torch.from_numpy(X_temp)
                y_temp = torch.from_numpy(y_temp)
                pos_temp = torch.from_numpy(pos_temp)
                
               
            X_temp = torch.cat((torch.full((self.RF2,),4, dtype=torch.int8), X_temp,
                                torch.full((self.RF2,),4, dtype=torch.int8)))
            pos_temp = pos_temp.clone() + self.RF2

            # mask gaps, deleting the parts of the genome where no CpG sites are labeled.
            mask = torch.ones_like(X_temp, dtype=torch.bool)
            for e, b in zip(pos_temp[1:][pos_temp[1:] - pos_temp[:-1] > self.RF],
                            pos_temp[:-1][pos_temp[1:] - pos_temp[:-1] > self.RF]):
                mask[torch.arange(b+self.RF2+1,e-self.RF2)] = False

            tmp = torch.zeros_like(X_temp, dtype=torch.int8)
            tmp[pos_temp.to(torch.long)] = 1
            tmp = tmp[mask]
            indices = torch.where(tmp)[0]
            X_temp = X_temp[mask]


            # skip the chromosome if it has less than segment_size CpG sites
            n_pos = len(pos_temp)
            if n_pos < self.segment_size:
                continue
            
            # prepare cuts that segment the genome & labels
            cuts_ = torch.arange(0,n_pos-self.segment_size+1,self.segment_size)
            cuts = torch.tensor([(indices[i],indices[i+self.segment_size-1]) for i in cuts_])
            cuts_ = torch.cat((cuts_, torch.tensor([n_pos-self.segment_size])))
            cuts = torch.cat((cuts, torch.tensor([(indices[-self.segment_size], indices[-1])])))

            batched_temp=[(X_temp[max(srt-self.RF2,0):stp+1+self.RF2],
                           y_temp[i:i+self.segment_size],
                           indices[i:i+self.segment_size]-indices[i]+self.RF2, 
                           pos_temp[i:i+self.segment_size]-pos_temp[i]) for i, (srt, stp) in zip(cuts_, cuts)]


            if self.val_keys is not None and chr_name in self.val_keys:
                val += batched_temp
            elif self.test_keys is not None and chr_name in self.test_keys:
                test += batched_temp
            elif self.fracs != [1,0,0]:
                random.shuffle(batched_temp)
                splits = np.cumsum(np.round(np.array(self.fracs)*len(batched_temp)).astype('int'))
                train += batched_temp[:splits[0]]
                val += batched_temp[splits[0]:splits[1]]
                test += batched_temp[splits[1]:]
            else:
                train += batched_temp

        self.train = CpGTransformerDataset(train, RF=self.RF,
                                           mask_percentage=self.mask_perc, 
                                           mask_random_percentage=self.mask_random_perc,
                                           resample_cells=self.resample)
        
        self.val = CpGTransformerDataset(val, RF=self.RF,
                                         mask_percentage=self.mask_perc, 
                                         mask_random_percentage=self.mask_random_perc,
                                         resample_cells=self.resample_val)

        self.test = test
        
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, num_workers=self.nw,
                                           batch_size=self.bsz, shuffle=True,
                                           pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, num_workers=self.nw,
                                           batch_size=self.bsz, shuffle=False,
                                           pin_memory=True)

    
    
class CpGTransformerDataset(torch.utils.data.Dataset):
    def __init__(self, split, RF=1001, mask_percentage=0.25,
                 mask_random_percentage=0.20, resample_cells=None):
        self.split = split
        
        RF2 = int((RF-1)/2)
        
        self.r = torch.arange(-RF2, RF2+1)
        self.k = RF
        
        # make a CDF of label distribution to sample from in randomizing:
        s = torch.stack([s[1] for s in split])
        s = s[s != -1]
        indices = torch.randperm(s.shape[0])[:2500]
        self.cdf = {'x': s[indices].sort().values, 'y': torch.linspace(0, 1, 2500)}
        
        self.mp = mask_percentage
        self.mrp = mask_random_percentage
        
        self.resample = resample_cells
        
        
    def __len__(self):
        return len(self.split)
    
    def __getitem__(self, index):
        x, y, ind, pos = self.split[index] # -1: unknown, 0: not methylated, 1: methylated
        
        x_windows = x[ind.unsqueeze(1).repeat(1,self.k)+self.r]
        cell_indices = torch.arange(y.shape[1])
            
        if self.resample:
            sample_indices = torch.randperm(cell_indices.shape[0])[:self.resample]
            cell_indices = cell_indices[sample_indices]
            y = y[:,sample_indices]
        
        y_orig = y+1
        seqlen, n_rep = y_orig.size()
        y_masked = y_orig.clone()
        
        nonzeros = y_masked.nonzero(as_tuple=False)
        n_permute = min(int(seqlen*self.mp), nonzeros.size(0))
        
        if self.mrp:
            n_mask, n_random = int(n_permute*(1-self.mrp)), math.ceil(n_permute*self.mrp)
            perm = torch.randperm(nonzeros.size(0))[:n_permute]
            nonzeros = nonzeros[perm]
            mask, rand = torch.split(nonzeros,[n_mask,n_random])
            
            y_masked[mask[:,0],mask[:,1]] = 0
            y_masked[rand[:,0],rand[:,1]] = sample_from_cdf(self.cdf, n_random)+1
        else:
            perm = torch.randperm(nonzeros.size(0))[:n_permute]
            nonzeros = nonzeros[perm]
            
            y_masked[nonzeros[:,0],nonzeros[:,1]] = 0
        
        return x_windows, y_orig, y_masked, pos, nonzeros, cell_indices
    

# Imputing dataset. Makes overlapping segments.
class CpGTransformerImputingDataModule(pl.LightningDataModule):
    def __init__(self, X, y, pos, segment_size=1024, RF=1001, RF_TF=81,
                 keys=None, n_workers=4):
        assert keys is None or type(keys) is list, 'keys should be None or list'
        super().__init__()
        
        self.X = X
        self.y = y
        self.pos = pos
        self.segment_size = segment_size
        self.RF = RF; self.RF2 = int((RF-1)/2)
        self.RF_TF = RF_TF; self.RF2_TF = int((RF_TF-1)/2)
        self.keys = keys
        self.nw = n_workers
        
    def setup(self, stage):
        
        self.datasets_per_chr = dict()
        
        iterate = self.keys if self.keys is not None else self.y.keys()
        for chr_name in iterate:
            y_temp = self.y[chr_name]
            X_temp = self.X[chr_name]
            pos_temp = self.pos[chr_name]

            if 'numpy' in str(type(X_temp)):
                X_temp = torch.from_numpy(X_temp)
                y_temp = torch.from_numpy(y_temp)
                pos_temp = torch.from_numpy(pos_temp)

            X_temp = torch.cat((torch.full((self.RF2,),4, dtype=torch.int8), X_temp,
                                torch.full((self.RF2,),4, dtype=torch.int8)))
            pos_temp = pos_temp.clone() + self.RF2

            # mask gaps, deleting the parts of the genome where no CpG sites are labeled.
            mask = torch.ones_like(X_temp, dtype=torch.bool)
            for e, b in zip(pos_temp[1:][pos_temp[1:] - pos_temp[:-1] > self.RF],
                            pos_temp[:-1][pos_temp[1:] - pos_temp[:-1] > self.RF]):
                mask[torch.arange(b+self.RF2+1,e-self.RF2)] = False

            tmp = torch.zeros_like(X_temp, dtype=torch.int8)
            tmp[pos_temp.to(torch.long)] = 1
            tmp = tmp[mask]
            indices = torch.where(tmp)[0]
            X_temp = X_temp[mask]


            n_pos = len(pos_temp)
            # prepare cuts that segment the genome & labels
            try:
                cuts_ = torch.arange(0,n_pos-self.segment_size,self.segment_size-self.RF2_TF*2)
                cuts = torch.tensor([(indices[i],indices[i+self.segment_size-1]) for i in cuts_])

                batched_temp=[(X_temp[max(srt-self.RF2,0):stp+1+self.RF2],
                           y_temp[i:i+self.segment_size],
                           indices[i:i+self.segment_size]-indices[i]+self.RF2, 
                           pos_temp[i:i+self.segment_size]-pos_temp[i]) for i, (srt, stp) in zip(cuts_, cuts)]

                cut_last_ = cuts_[-1]+self.segment_size-self.RF2_TF*2
                cut_last = torch.tensor([indices[cut_last_],indices[-1]])
                srt, stp = cut_last

                batched_temp += [(X_temp[max(srt-self.RF2,0):stp+1+self.RF2],
                           y_temp[cut_last_:],
                           indices[cut_last_:]-indices[cut_last_]+self.RF2, 
                           pos_temp[cut_last_:]-pos_temp[cut_last_])]
            except:
                cut_last_ = 0
                cut_last = torch.tensor([indices[cut_last_],indices[-1]])
                srt, stp = cut_last

                batched_temp = [(X_temp[max(srt-self.RF2,0):stp+1+self.RF2],
                           y_temp[cut_last_:],
                           indices[cut_last_:]-indices[cut_last_]+self.RF2, 
                           pos_temp[cut_last_:]-pos_temp[cut_last_])]

        
            self.datasets_per_chr[chr_name] = torch.utils.data.DataLoader(
                ImputingDataset(batched_temp, RF=self.RF),
                num_workers = self.nw, shuffle=False, pin_memory=True)
        
# Imputing dataset. Makes overlapping segments.
class ImputingDataset(torch.utils.data.Dataset):
    def __init__(self, split, RF=1001):
        self.split = split
        
        RF2 = int((RF-1)/2)
        self.r = torch.arange(-RF2, RF2+1)
        self.k = RF
        
    def __len__(self):
        return len(self.split)
    
    def __getitem__(self, index):
        x, y, ind, pos = self.split[index] 
        
        y += 1
        
        x_windows = x[ind.unsqueeze(1).repeat(1,self.k)+self.r]
        cell_indices = torch.arange(y.shape[1])
        
        return x_windows, y, pos, cell_indices    

    
    
    
# Exhaustive testing. Only used in benchmarking, not in practical imputation.
class ExhaustiveBenchmarkDataset(torch.utils.data.Dataset):
    """
    Works only for one chromosome at a time.
    Meaning for every chromosome tested this way you need new object of this type.
    """
    def __init__(self, DNA_embeddings, y, pos, window=81, RF=1001,
                 output_rows=None, input_rows=None, filter_columns=None):
        if input_rows is not None and output_rows is not None:
            assert set(output_rows).intersection(set(input_rows)) == set(output_rows)
        
        RF2 = int((RF-1)/2)
        
        if 'numpy' in str(type(y)):
            y = torch.from_numpy(y)
            pos = torch.from_numpy(pos)

        pos = pos.clone() + RF2
        
        self.data = {'DNA': DNA_embeddings, 'y': y, 'pos': pos}
        
        self.input_rows = input_rows
        if self.input_rows is not None:
            self.input_rows = torch.tensor(self.input_rows)
            self.data['y'] = self.data['y'][:,self.input_rows]
            if filter_columns:
                indices = (self.data['y'] != -1).sum(1) != 0
                self.data['y'] = self.data['y'][indices]
                self.data['DNA'] = self.data['DNA'][indices]
                self.data['pos'] = self.data['pos'][indices]
                
        self.output_rows = output_rows
        if self.output_rows is not None:
            self.output_rows = torch.tensor(self.output_rows)
            if self.input_rows is not None:
                self.output_rows = torch.tensor([torch.where(self.input_rows == i) for i in self.output_rows]).view(-1)
        
        
        self.list_indices = (self.data['y']!=-1).nonzero(as_tuple=False)
        self.w = int((window-1)/2)
        
        self.RF2 = RF2
        
        if self.output_rows is not None:
            self.list_indices = self.list_indices[(self.list_indices[:,1][...,None] == self.output_rows).any(-1)]
        
        self.edges = self.list_indices[self.list_indices[:,0]<self.w]
        self.edges = torch.cat([self.edges, self.list_indices[self.list_indices[:,0]>=self.data['y'].shape[0]-self.w]])
        
        self.list_indices = self.list_indices[self.list_indices[:,0]>=self.w]
        self.list_indices = self.list_indices[self.list_indices[:,0]<self.data['y'].shape[0]-self.w]
        
    def __len__(self):
        return self.list_indices.shape[0]
    
    def __getitem__(self, index):
        i, index_label = self.list_indices[index]
        
        
        pos = self.data['pos'][max(0,i-self.w):i+self.w+1]
        y = self.data['y'][max(0,i-self.w):i+self.w+1]
        DNA = self.data['DNA'][max(0,i-self.w):i+self.w+1]
        
        if self.input_rows is not None:
            cell_indices = self.input_rows.clone()
        else:
            cell_indices = torch.arange(y.shape[1])
        
        
        y_input = y+1
        y_input[i-max(0,i-self.w), index_label] = 0 
            
        return DNA, y_input, pos, (i, index_label), cell_indices
    
    def edge_getitem(self, index):
        i, index_label = self.edges[index]
        
        
        pos = self.data['pos'][max(0,i-self.w):i+self.w+1]
        y = self.data['y'][max(0,i-self.w):i+self.w+1]
        DNA = self.data['DNA'][max(0,i-self.w):i+self.w+1]
        
        if self.input_rows is not None:
            cell_indices = self.input_rows.clone()
        else:
            cell_indices = torch.arange(y.shape[1])
        
        
        y_input = y+1
        y_input[i-max(0,i-self.w), index_label] = 0 
            
        return DNA, y_input, pos, (i, index_label), cell_indices
    
# Exhaustive testing. Only used in benchmarking, not in practical imputation.
class ExhaustiveBenchmarkDNAEmbedding(torch.utils.data.Dataset):
    """
    Works only for one chromosome at a time
    Meaning for every chromosome tested this way you need new object of this type.
    """
    def __init__(self, X, pos, RF=1001):
        RF2 = int((RF-1)/2)
        
        if 'numpy' in str(type(X)):
            X = torch.from_numpy(X) 
            pos = torch.from_numpy(pos)

        X = torch.cat((torch.full((RF2,),4, dtype=torch.int8), X,
                    torch.full((RF2,),4, dtype=torch.int8)))
        pos = pos.clone() + RF2
        
        mask = torch.ones_like(X, dtype=torch.bool)
        for e, b in zip(pos[1:][pos[1:] - pos[:-1] > RF],
                        pos[:-1][pos[1:] - pos[:-1] > RF]):
            mask[torch.arange(b+RF2+1,e-RF2)] = False

        tmp = torch.zeros_like(X, dtype=torch.int8)
        tmp[pos.to(torch.long)] = 1
        tmp = tmp[mask]
        indices = torch.where(tmp)[0]
        X = X[mask]
        
        self.data = {'X': X, 'indices': indices}
        self.list_indices = torch.arange(pos.shape[0])

        self.r = torch.arange(-RF2, RF2+1)
        self.k = RF
        self.RF2 = RF2
        
    def __len__(self):
        return self.list_indices.shape[0]
    
    def __getitem__(self, index):
        i = self.list_indices[index]
        
        ind = self.data['indices'][i].unsqueeze(0)
        
        #x = [max(0,ind[0]-self.RF2):ind[-1]+1+self.RF2]
        #ind = ind - self.data['indices'][max(0,i-self.w)]+self.RF2
        x_windows = self.data['X'][ind.repeat(1,self.k)+self.r]
            
        return x_windows
    
    
# DeepCpG    
class DeepCpGDataModule(pl.LightningDataModule):
    def __init__(self,X, y, pos, RF=1001, fracs=[1,0,0],
                 val_keys=None, test_keys=None,
                 batch_size=128, n_workers=4,
                 window=25, max_dist=25000, batch_pos=False,
                 batch_index=False):
        super().__init__()
        self.X = X
        self.y = y
        self.pos = pos
        self.RF = RF
        self.RF2 = int((RF-1)/2)
        self.fracs = fracs
        self.val_keys = val_keys
        self.test_keys = test_keys
        self.r = torch.arange(-self.RF2, self.RF2+1)
        self.k = RF 
        self.bsz = batch_size
        self.nw = n_workers
        self.w = window
        self.max_dist = torch.tensor([max_dist],dtype=torch.float)
        self.batch_pos = batch_pos # include position in batches, used in benchmarking
        self.batch_index = batch_index # include index in dataset in batches, used in imputation
    def setup(self,stage):
        data = {'ind': [], 'X': [], 'repy': [],'reppos': [],
                'pos': [], 'y': [], 'ix_rep': [], 'extra_rep': []}
        
        train = []; val = []; test = []
        for ix_chr, chr_name in enumerate(self.y.keys()):
            X_temp = self.X[chr_name]
            y_temp = self.y[chr_name]
            pos_temp = self.pos[chr_name]

            if 'numpy' in str(type(X_temp)):
                X_temp = torch.from_numpy(X_temp)
                y_temp = torch.from_numpy(y_temp)
                pos_temp = torch.from_numpy(pos_temp)

            X_temp = torch.cat((torch.full((self.RF2,),4, dtype=torch.int8), X_temp,
                                torch.full((self.RF2,),4, dtype=torch.int8)))
            pos_temp += self.RF2

            # mask gaps:
            mask = torch.ones_like(X_temp, dtype=torch.bool)
            for e, b in zip(pos_temp[1:][pos_temp[1:] - pos_temp[:-1] > self.RF],
                            pos_temp[:-1][pos_temp[1:] - pos_temp[:-1] > self.RF]):
                mask[torch.arange(b+self.RF2+1,e-self.RF2)] = False

            tmp = torch.zeros_like(X_temp, dtype=torch.int8)
            tmp[pos_temp.to(dtype=torch.long)] = 1
            tmp = tmp[mask]
            indices = torch.where(tmp)[0]
            X_temp = X_temp[mask]

            n_cpgs = pos_temp.shape[0]
            batched_temp = [(ix_chr, ix_pos) for ix_pos in range(n_cpgs)]

            replicates_y = [torch.cat((torch.tensor([.5]*self.w, dtype=torch.float16),
                                       p[p!=-1].to(dtype=torch.float16),
                                       torch.tensor([.5]*self.w, dtype=torch.float16))) for p in y_temp.T]
            
            replicates_pos = [torch.cat((torch.tensor([pos_temp[0]-self.max_dist*2]*self.w).to(dtype=pos_temp.dtype),
                                    pos_temp[p!=-1],
                                    torch.tensor([pos_temp[-1]+self.max_dist*2]*self.w).to(dtype=pos_temp.dtype))) for p in y_temp.T]

            n_rep = len(replicates_y)
            replicates_y = nn.utils.rnn.pad_sequence(replicates_y, batch_first=True)
            replicates_pos = nn.utils.rnn.pad_sequence(replicates_pos, batch_first=True)
            ix_counter = torch.tensor([self.w]*n_rep)

            ix = torch.zeros_like(y_temp).to(dtype=pos_temp.dtype)
            extra = torch.zeros_like(y_temp).to(dtype=torch.bool)

            for row_positions, p in enumerate(pos_temp):
                ix[row_positions,:]=ix_counter.clone()
                matches = y_temp[row_positions]!=-1
                extra[row_positions,:]=matches
                ix_counter+=matches

                if row_positions % 5000 == 0:
                        print(chr_name, np.round(row_positions/n_cpgs*100,2),'%\t\t\t', end='\r')
            print(chr_name, np.round((row_positions+1)/n_cpgs*100,2),'%\t\t\t')


            if self.val_keys is not None and chr_name in self.val_keys:
                val += batched_temp
            elif self.test_keys is not None and chr_name in self.test_keys:
                test += batched_temp
            elif self.fracs != [1,0,0]:
                random.shuffle(batched_temp)
                splits = np.cumsum(np.round(np.array(self.fracs)*len(batched_temp)).astype('int'))
                train += batched_temp[:splits[0]]
                val += batched_temp[splits[0]:splits[1]]
                test += batched_temp[splits[1]:]
            else:
                train += batched_temp
        
            data['pos'].append(pos_temp)
            data['X'].append(X_temp)
            data['y'].append(y_temp)
            data['ind'].append(indices)
            data['repy'].append(replicates_y)
            data['reppos'].append(replicates_pos)
            data['ix_rep'].append(ix.to(dtype=torch.int))
            data['extra_rep'].append(extra)
        
        
        self.data = data
        self.train = DeepCpGDataset(train, self.data, self.RF, self.w, self.max_dist,
                                    batch_pos=self.batch_pos, batch_index=self.batch_index)
        self.val = DeepCpGDataset(val, self.data, self.RF, self.w, self.max_dist,
                                  batch_pos=self.batch_pos, batch_index=self.batch_index)
        self.test = test
        
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, num_workers=self.nw, 
                                           batch_size=self.bsz, shuffle=True, pin_memory=True)
        
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, num_workers=self.nw,
                                           batch_size=self.bsz, shuffle=False, pin_memory=True)
    
class DeepCpGDataset(torch.utils.data.Dataset):
    def __init__(self, list_indices, data, RF, window, max_dist, batch_pos=False, batch_index=False):
        self.list_indices = list_indices
        self.data = data
        self.RF = RF; self.RF2 = int((RF-1)/2)
        self.r = torch.arange(-self.RF2, self.RF2+1)
        self.w = window
        self.max_dist = torch.tensor([max_dist],dtype=torch.float)
        self.b = torch.arange(-window,0)
        self.a = torch.arange(0,window)
        self.batch_pos = batch_pos
        self.batch_index = batch_index
    def __len__(self):
        return len(self.list_indices)

    def __getitem__(self, index):
        ix_chr, ix_pos = self.list_indices[index]
        
        X = self.data['X'][ix_chr]
        ind = self.data['ind'][ix_chr][ix_pos]
        DNA = X[ind.repeat(self.RF)+self.r]
        
        pos = self.data['pos'][ix_chr][ix_pos]
        ix = self.data['ix_rep'][ix_chr][ix_pos]
        extra = self.data['extra_rep'][ix_chr][ix_pos]
        n_rep = ix.size(0)
        indices = torch.cat(((ix.expand(self.w,-1).T+self.b),
                   ((ix+extra).expand(self.w,-1).T+self.a)),1)
        CpG = torch.stack((torch.gather(self.data['repy'][ix_chr],1, indices).to(torch.float),
                     torch.gather(self.data['reppos'][ix_chr],1, indices).to(torch.float)),2).view(n_rep,-1)        
        
        CpG[:,1:self.w*2:2] = torch.min((pos-CpG[:,1:self.w*2:2]).to(dtype=torch.float),
                                       self.max_dist)/self.max_dist
        CpG[:,self.w*2+1::2] = torch.min((CpG[:,self.w*2+1::2]-pos).to(dtype=torch.float),
                                 self.max_dist)/self.max_dist
       
        y = self.data['y'][ix_chr][ix_pos]
        if not self.batch_pos:
            if not self.batch_index:
                return DNA, CpG, y
            else:
                return DNA, CpG, y, (ix_chr, ix_pos)
        else:
            if not self.batch_index:
                return DNA, CpG, y, pos
            else:
                return DNA, CpG, y, pos, (ix_chr, ix_pos)

        
        
# CaMelia
class CaMeliaPreprocessor():
    def __init__(self, X, y, pos, val_keys=['chr5'], test_keys=['chr10'], fracs=[1,0,0]):
        assert len(fracs)==3,'length of fractions should be 3 for train/val/test'
        assert sum(fracs)==1, 'Sum of train/val/test fractions should be one.'
        assert val_keys is None or type(val_keys) is list, 'val_keys should be None or list'
        assert test_keys is None or type(test_keys) is list, 'test_keys should be None or list'
        if val_keys is not None and test_keys is not None:
            assert set(val_keys) & set(test_keys) == set(), 'No overlap allowed between val_keys & test_keys'
        
        
        self.X = X
        self.y = y
        self.pos = pos
        self.val_keys = val_keys
        self.test_keys = test_keys
        self.fracs = fracs
        
        nuctoix = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4,
                   'M': 5, 'R': 6, 'W': 7, 'S': 8, 'Y': 9,
                   'K': 10, 'V': 11, 'H': 12, 'D': 13,
                   'B': 14, 'X': 15}
        ixtonuc = {v:k for k,v in nuctoix.items()}
        self.ixtonuc_list = np.array([ixtonuc[i] for i in range(16)])
        
    def DNA_feature(self, chr_key, cell_index, n=10, whole_genome=False):
        if not whole_genome:
            positions_to_use = self.y[chr_key][:,cell_index] != -1
        else:
            positions_to_use = np.full(self.y[chr_key][:,cell_index].shape, True)
        pos_subset = self.pos[chr_key][positions_to_use]+n
        range_ = np.concatenate([np.arange(-n,0),np.arange(1,n+1)])
        
        X_padded = np.concatenate([np.array([4]*n), self.X[chr_key], np.array([4]*n)])
        
        features = X_padded[pos_subset[:,None]+range_]
        return self.ixtonuc_list[features] # convert indices to string
    
    def local_cell_similarity_feature(self, chr_key, cell_index, threshold=0.8, whole_genome=False):
        if not whole_genome:
            target_sites = self.y[chr_key][:,cell_index] != -1
        else:
            target_sites = np.full(self.y[chr_key][:,cell_index].shape, True)

        target_pos = self.pos[chr_key][target_sites]


        other_cells_select = np.full(self.y[chr_key].shape[1], True)
        other_cells_select[cell_index] = False


        y_all_cell = self.y[chr_key][:, cell_index]
        y_all_other_cells = self.y[chr_key][:, other_cells_select]

        observed_cell_ix = y_all_cell != -1
        observed_other_cells_ix = y_all_other_cells != -1


        pos_observed_target = np.concatenate([self.pos[chr_key][observed_cell_ix],np.array([-1])])

        pos_observed_cells = []
        y_observed_cells = []
        for i in range(len(other_cells_select)-1):
            slct_observed = observed_other_cells_ix[:,i] & target_sites
            pos_observed_cells.append(self.pos[chr_key][slct_observed])
            y_observed_cells.append(y_all_other_cells[slct_observed,i])

        pos_observed_cells_padded = np.full((len(pos_observed_cells), max([len(a) for a in pos_observed_cells])+1), -1)
        for ix, p in enumerate(pos_observed_cells):
            pos_observed_cells_padded[ix,:len(p)] = p

        y_observed_cells_padded = np.full((len(y_observed_cells), max([len(a) for a in y_observed_cells])+1), -1)
        for ix, p in enumerate(y_observed_cells):
            y_observed_cells_padded[ix,:len(p)] = p        


        observed_other_cells_in_common = np.expand_dims(observed_cell_ix,1) & observed_other_cells_ix
        pos_pairs = []
        y_other_cells_pairs = []
        y_cell_pairs = []
        for i in range(len(other_cells_select)-1):
            pos_pairs.append(self.pos[chr_key][observed_other_cells_in_common[:,i]])
            y_other_cells_pairs.append(y_all_other_cells[observed_other_cells_in_common[:,i],i])
            y_cell_pairs.append(y_all_cell[observed_other_cells_in_common[:,i]])

        pos_pairs_padded = np.full((len(pos_pairs), max([len(a) for a in pos_pairs])+20),
                                      max([max(a) for a in pos_pairs if len(a)>0])+1)

        for ix, p in enumerate(pos_pairs):
            pos_pairs_padded[ix,10:10+len(p)] = p
        pos_pairs_padded[:,:10] = 0

        y_other_cells_pairs_padded = np.full((len(pos_pairs), max([len(a) for a in pos_pairs])+20), -1)
        for ix, p in enumerate(y_other_cells_pairs):
            y_other_cells_pairs_padded[ix,10:10+len(p)] = p

        y_cell_pairs_padded = np.full((len(pos_pairs), max([len(a) for a in pos_pairs])+20), -2)
        for ix, p in enumerate(y_cell_pairs):
            y_cell_pairs_padded[ix,10:10+len(p)] = p

        #fixed
        range_1 = np.concatenate([np.arange(-10,0), np.arange(1,11)])
        range_2 = np.arange(-10,10)
        cell_indexer = np.arange(pos_pairs_padded.shape[0])
        #changes throughout iteration
        pos_indexer = np.array([10]*pos_pairs_padded.shape[0])
        pos_indexer2 = np.array([0]*pos_pairs_padded.shape[0])
        ticker_pos_obs_t = 0

        features = np.full([len(target_pos)], np.nan)

        for i in range(len(target_pos)):

            matches = pos_observed_cells_padded[cell_indexer, pos_indexer2] == target_pos[i]
            is_observed = target_pos[i] == pos_observed_target[ticker_pos_obs_t]
            range_ = range_1 if is_observed else range_2
            row_to_select = np.where(matches)[0]
            col_to_select = pos_indexer[row_to_select]
            L_k = y_other_cells_pairs_padded[row_to_select[:,None],np.expand_dims(col_to_select,1)+range_]
            L_target = y_cell_pairs_padded[row_to_select[:,None],np.expand_dims(col_to_select,1)+range_]
            PS_k = (L_k == L_target).sum(-1)/20
            PS_k_threshold = PS_k > threshold
            if PS_k_threshold.sum() > 0:
                CpG_k_target = y_observed_cells_padded[row_to_select,pos_indexer2[row_to_select]]
                features[i] = np.sum(PS_k[PS_k_threshold]*np.log2(CpG_k_target[PS_k_threshold]+1.01))/PS_k_threshold.sum()

            if is_observed:
                pos_indexer += matches
                ticker_pos_obs_t += 1
            pos_indexer2 += matches


            if i % 10000 == 0:
                print('Progress encoding local sim. feat. for', chr_key, '...:', np.round(i/len(target_pos)*100,2),'%', end='\r')
        print('Progress encoding local sim. feat. for', chr_key, '...:', np.round((i+1)/len(target_pos)*100,2),'%', end='\r')
        print()

        return features.reshape(-1,1)
    
    def neighbor_methylation_feature(self, chr_key, cell_index, whole_genome=False):
        if not whole_genome:
            positions_to_use = self.y[chr_key][:,cell_index] != -1
        else:
            positions_to_use = np.full(self.y[chr_key][:,cell_index].shape, True)
            
        pos_subset = self.pos[chr_key][positions_to_use]
        # for the edge cases we fill in a fake position corresponding to the max position that will be encountered in their windows
        pos_subset = np.concatenate((np.full([10], pos_subset[0]*2-pos_subset[10]),
                                     pos_subset, np.full([10], pos_subset[-1]*2-pos_subset[11])))
        y_subset = self.y[chr_key][positions_to_use, cell_index]
        y_subset = np.concatenate((np.full([10], 0.5), y_subset, np.full([10], 0.5)))

        rolled_pos = np.array([np.roll(pos_subset, i) for i in range(10,-11,-1)])[:,10:-10]
        rolled_y = np.array([np.roll(y_subset, i) for i in range(10,-11,-1)])[:,10:-10]

        rolled_dist = np.abs(rolled_pos - rolled_pos[10])
        rolled_dist_relative = (1-(rolled_dist/rolled_dist.max(0)))
        features = np.log2(rolled_y+1.01)*rolled_dist_relative

        features = features[[True]*10+[False]+[True]*10].T # leave the middle row out: the row of the location itself.
        return features
    def __call__(self, cell_index, neigh=True, local=True, DNA=True, threshold=0.8, whole_genome=False):
        X_train = pd.DataFrame(); y_train = np.empty(0, dtype='int8')
        X_val = pd.DataFrame(); y_val = np.empty(0, dtype='int8')
        X_test = pd.DataFrame(); y_test = np.empty(0, dtype='int8')
        pos_val = np.empty(0, dtype='int32'); pos_test = np.empty(0, dtype='int32')
        
        for chr_key in self.y.keys():
            indices_cell_index = self.y[chr_key][:,cell_index] != -1
            y_cell = self.y[chr_key][indices_cell_index, cell_index]
            
            features_list = []
            if DNA:
                DNA_feat = self.DNA_feature(chr_key, cell_index, whole_genome=whole_genome)
                features_list.append(pd.DataFrame(DNA_feat, columns=['DNA'+str(i) for i in range(20)]).astype('category'))
            if local:
                local_feat = self.local_cell_similarity_feature(chr_key, cell_index,threshold=threshold, whole_genome=whole_genome)
                features_list.append(pd.DataFrame(local_feat, columns = ['local']))
            if neigh:
                neigh_feat = self.neighbor_methylation_feature(chr_key, cell_index, whole_genome=whole_genome)
                features_list.append(pd.DataFrame(neigh_feat, columns=['neigh'+str(i) for i in range(20)]))
            dataframe = pd.concat(features_list,axis=1)
            pos_cell = self.pos[chr_key][indices_cell_index]
                
            if self.val_keys is not None and chr_key in self.val_keys:
                X_val = pd.concat([X_val, dataframe], ignore_index=True, axis=0)
                y_val = np.concatenate([y_val, y_cell])
                pos_val = np.concatenate([pos_val, pos_cell])
                
                
            elif self.test_keys is not None and chr_key in self.test_keys:
                X_test = pd.concat([X_test, dataframe], ignore_index=True, axis=0)
                y_test = np.concatenate([y_test, y_cell])
                pos_test = np.concatenate([pos_test, pos_cell])
                
            elif self.fracs != [1,0,0]:
                ix = np.arange(len(y_cell))
                np.random.shuffle(ix)
                dataframe, y_cell, pos_cell = dataframe.iloc[ix].reset_index(drop=True), y_cell[ix], pos_cell[ix]
                splits = np.cumsum(np.round(np.array(self.fracs)*len(batched_temp)).astype('int'))
                train += batched_temp[:splits[0]]
                
                X_train = pd.concat([X_train, dataframe.iloc[:splits[0]]], ignore_index=True, axis=0)
                y_train = np.concatenate([y_train, y_cell[:splits[0]]])
                
                X_val = pd.concat([X_val, dataframe.iloc[splits[0]:splits[1]]], ignore_index=True, axis=0)
                y_val = np.concatenate([y_val, y_cell[splits[0]:splits[1]]])
                pos_val = np.concatenate([pos_val, pos_cell[splits[0]:splits[1]]])
                
                X_test = pd.concat([X_test, dataframe.iloc[splits[1]:]], ignore_index=True, axis=0)
                y_test = np.concatenate([y_test, y_cell[splits[1]:]])
                pos_test = np.concatenate([pos_test, pos_cell[splits[1]:]])
                
            else:
                X_train = pd.concat([X_train, dataframe], ignore_index=True, axis=0)
                y_train = np.concatenate([y_train, y_cell])
            
        return X_train, X_val, X_test, y_train, y_val, y_test, pos_val, pos_test