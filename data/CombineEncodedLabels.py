import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Combine encoded labels from pos.npz and y.npz files')

parser.add_argument('--y_files', nargs="+", type=str,
                    help='y files')
parser.add_argument('--pos_files', nargs="+", type=str,
                    help='pos files')
parser.add_argument('--y_outFile', type=str,
                    help='output file to save encoded labels in.')
parser.add_argument('--pos_outFile', type=str,
                    help='output file to save encoded positions of labels in.')
args = parser.parse_args()

print('Reading data ...')

y_files = sorted(args.y_files)
pos_files = sorted(args.pos_files)

print('Ordering of cells:')
for a,b in zip(y_files, pos_files):
    a = a.split('/')[-1].split('_')[1].split('.')[0]
    b = b.split('/')[-1].split('_')[1].split('.')[0]
    
    assert a == b, 'Ordering is wrong.'
    print(a, end=',')
print(" \n")

ys = [np.load(d) for d in y_files]
poss = [np.load(d) for d in pos_files]

pos_combined = {}
ys_combined = {}

for chrom in ys[0].keys():
    print('Combining', chrom, '...')
    pos_combined_chrom = np.unique(np.hstack([p[chrom] for p in poss])) 
    pos_combined[chrom] = pos_combined_chrom

    ys_combined_chrom = np.full((pos_combined_chrom.shape[0], len(ys)), -1, dtype='int8')
    
    for ix, p in enumerate(poss):
        
        indices = np.in1d(pos_combined_chrom, p[chrom])
        ys_combined_chrom[indices, ix] = ys[ix][chrom]
        
    ys_combined[chrom] = ys_combined_chrom

    
print('Writing combined files ...')
np.savez_compressed(args.y_outFile, **ys_combined)
np.savez_compressed(args.pos_outFile, **pos_combined)