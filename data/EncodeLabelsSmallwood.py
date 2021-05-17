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

dat_plus = dat[dat[2]=='+']
dat_minus = dat[dat[2]=='-']
assert ((dat_minus[1]-1).values != dat_plus[1].values).sum()==0, 'No 100\% overlap between CpG sites for + in - strands in data'

count_meth = dat_plus[3].values+dat_minus[3].values
count_unmeth = dat_plus[4].values+dat_minus[4].values
to_fill_df = dat_plus[[0,1]]

filter_ = count_meth+count_unmeth != 0

count_meth = count_meth[filter_]
count_unmeth = count_unmeth[filter_]
to_fill_df = to_fill_df[filter_]

to_fill_df.loc[:,2] = (count_meth/(count_meth+count_unmeth)>=0.5).astype('int8')
to_fill_df.loc[:,3] = count_meth.astype('uint16')
to_fill_df.loc[:,4] = count_unmeth.astype('uint16')

X_encoded = np.load(args.EncodedGenome)

y_encoded = {}
pos_encoded = {}

for chrom_name in chroms:
    print('Encoding',chrom_name,'...')
    X_chrom = X_encoded[chrom_name]
    indices = np.where(X_chrom==2)[0]

    dat_chrom = to_fill_df[to_fill_df[0] == chrom_name[3:]]

    dat_subset = dat_chrom[np.in1d(dat_chrom[1].values-1, indices)]
    y_encoded[chrom_name] = dat_subset[2].values.astype('int8')
    pos_encoded[chrom_name] = (dat_subset[1].values-1).astype('int32')


np.savez_compressed(args.y_outFile, **y_encoded)
np.savez_compressed(args.pos_outFile, **pos_encoded)
