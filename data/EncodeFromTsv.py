import argparse
import pandas as pd
import numpy as np
import re

parser = argparse.ArgumentParser(description='Encode CpG labels from Wig file into compact format')

parser.add_argument('dataFile', type=str, metavar='<.tsv file>',
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
parser.add_argument('--zero_indexed', action='store_true',
                    help='Add this flag if your dataFile positions are already zero indexed.')

args = parser.parse_args()

chroms = args.chroms
if args.prepend_chr:
    chroms = ["chr" + c for c in chroms]

print('Reading data ...')
dat = pd.read_csv(args.dataFile, sep='\t', header=None,
                 dtype={0:'string'})

if not args.zero_indexed:
    dat[1] = dat[1] - 1

for chrom_name in chroms:
    print('Encoding',chrom_name,'...')
    X_chrom = X_encoded[chrom_name]
    indices = np.where(X_chrom==2)[0]

    dat_chrom = dat[dat[0] == chrom_name]

    dat_subset = dat_chrom[np.in1d(dat_chrom[1].values, indices)]
    
    y_encoded[chrom_name] = dat_subset.iloc[:,2:].values.astype('int8')
    pos_encoded[chrom_name] = (dat_subset[1].values).astype('int32')


np.savez_compressed(args.y_outFile, **y_encoded)
np.savez_compressed(args.pos_outFile, **pos_encoded)
