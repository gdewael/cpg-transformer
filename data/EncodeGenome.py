import argparse
from Bio import SeqIO
import numpy as np


parser = argparse.ArgumentParser(description='Encode genome to a compact format.')

parser.add_argument('fastaFile', type=str, metavar='<.fa file>',
                    help='FASTA file with genome sequences')
parser.add_argument('outFile', type=str, metavar='<.npz file>',
                    help='output file to save encoded genome in.')
parser.add_argument('--chroms', nargs="+", type=str, required=True,
                    help='ordering of chromosomes in the fasta file')
parser.add_argument('--prepend_chr', action='store_true',
                    help='whether to prepend the str "chr" to the names of chromosomes given in --chroms.')


args = parser.parse_args()

chroms = args.chroms

if args.prepend_chr:
    chroms = ["chr" + c for c in chroms]

seq = [seq for seq in SeqIO.parse(args.fastaFile, "fasta")]
    
    
nuctoix = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4,
       'M': 5, 'R': 6, 'W': 7, 'S': 8, 'Y': 9,
       'K': 10, 'V': 11, 'H': 12, 'D': 13,
       'B': 14, 'X': 15}

X_encoded = {}
for chrom_seq, chrom_name in zip(seq, chroms):
    print('Encoding',chrom_name,'...')
    X_encoded[chrom_name] = np.array([nuctoix[i] for i in chrom_seq.seq.upper()], dtype='int8')
    
np.savez_compressed(args.outFile, **X_encoded)