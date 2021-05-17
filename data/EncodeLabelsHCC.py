import argparse
import pandas as pd
import numpy as np
import re

parser = argparse.ArgumentParser(description='Encode CpG labels from Wig file into compact format')

parser.add_argument('dataFile', type=str, metavar='<.txt file>',
                    help='data file')
parser.add_argument('EncodedGenome', type=str, metavar='<.npz file>',
                    help='Encoded genome from EncodeGenome.py.')
parser.add_argument('y_outFile', type=str, metavar='<.npz file>',
                    help='output file to save encoded labels in.')
parser.add_argument('pos_outFile', type=str, metavar='<.npz file>',
                    help='output file to save encoded positions of labels in.')
parser.add_argument('--chroms', nargs="+", type=str, required=True,
                    help='ordering of chromosomes in the fasta file')
parser.add_argument('--prepend_chr', action='store_true',
                    help='whether to prepend the str "chr" to the names of chromosomes given in --chroms.')

args = parser.parse_args()

chroms = args.chroms
if args.prepend_chr:
    chroms = ["chr" + c for c in chroms]

print('Reading data ...')
dat = pd.read_csv(args.dataFile, sep='\t', header=None,
                  dtype={0:'string', 2:'string',3:'string',8:'string', 9:'string'})

X_encoded = np.load(args.EncodedGenome)

y_encoded = {}
pos_encoded = {}
for chrom_name in chroms:
    print('Encoding',chrom_name,'...')
    X_chrom = X_encoded[chrom_name]
    indices = np.where(X_chrom==2)[0]
    
    dat_subset = dat[dat[0] == chrom_name]
    dat_subset_plus = dat_subset[dat_subset[3] == "+"][[1,4,5,6]]
    dat_subset_min = dat_subset[dat_subset[3] == "-"][[1,4,5,6]]
    dat_subset_min[1] -= 1
    
    
    overlapping_plus = np.isin(dat_subset_plus[1], dat_subset_min[1])
    overlapping_min = np.isin(dat_subset_min[1], dat_subset_plus[1])
    dat_comb = np.concatenate((dat_subset_plus[overlapping_plus][1].values.reshape(-1,1), 
                               dat_subset_plus[overlapping_plus][[4,5,6]].values+\
                               dat_subset_min[overlapping_min][[4,5,6]].values),1)

    dat_all = np.concatenate((dat_comb, dat_subset_plus[~overlapping_plus][[1,4,5,6]].values,
                    dat_subset_min[~overlapping_min][[1,4,5,6]].values))

    dat_all = dat_all[np.argsort(dat_all[:,0])]
    
    count_meth_chrom = dat_all[:,2].astype('uint16')
    count_unmeth_chrom = dat_all[:,3].astype('uint16')
    label_chrom = (count_meth_chrom/(count_meth_chrom+count_unmeth_chrom)>=0.5).astype('int8')
    
    
    subset_ind_C = np.in1d(dat_all[:,0]-1, indices)
    
    count_meth_chrom_C = count_meth_chrom[subset_ind_C]
    count_unmeth_chrom_C = count_unmeth_chrom[subset_ind_C]
    
    
    y_encoded[chrom_name] = label_chrom[subset_ind_C].astype('int8')
    pos_encoded[chrom_name] = (dat_all[:,0]-1)[subset_ind_C].astype('int32')

np.savez_compressed(args.y_outFile, **y_encoded)
np.savez_compressed(args.pos_outFile, **pos_encoded)