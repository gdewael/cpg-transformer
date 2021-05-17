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
                 dtype={0:'string'})

count_meth = dat[2].values
count_unmeth = dat[3].values - count_meth

filter_ = count_meth+count_unmeth != 0

count_meth = count_meth[filter_].astype('uint16')
count_unmeth = count_unmeth[filter_].astype('uint16')
dat = dat[filter_]

label = (count_meth/(count_meth+count_unmeth)>=0.5).astype('int8')

X_encoded = np.load(args.EncodedGenome)

y_encoded = {}
pos_encoded = {}
for chrom_name in chroms:
    print('Encoding',chrom_name,'...')
    X_chrom = X_encoded[chrom_name]
    indices = np.where(X_chrom==2)[0]
    
    subset_ind = (dat[0] == chrom_name).to_numpy(dtype='bool')
    dat_chrom = dat[subset_ind]
    count_meth_chrom = count_meth[subset_ind]
    count_unmeth_chrom = count_unmeth[subset_ind]
    label_chrom = label[subset_ind]
    
    subset_ind_C = np.in1d(dat_chrom[1].values-1, indices)
    
    dat_chrom_C = dat_chrom[subset_ind_C]
    count_meth_chrom_C = count_meth_chrom[subset_ind_C]
    count_unmeth_chrom_C = count_unmeth_chrom[subset_ind_C]
    label_chrom_C = label_chrom[subset_ind_C]
    
    
    y_encoded[chrom_name] = label_chrom_C.astype('int8')
    pos_encoded[chrom_name] = (dat_chrom_C[1].values-1).astype('int32')

np.savez_compressed(args.y_outFile, **y_encoded)
np.savez_compressed(args.pos_outFile, **pos_encoded)