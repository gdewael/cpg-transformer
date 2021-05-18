import numpy as np
from argparse import ArgumentParser

parser.add_argument('model', choices=['cpg_transformer', 'deepcpg', 'camelia'],
                    help='Which model type to use for imputation.')

optional_parse = parser.add_argument_group('General optional arguments.')


cpgtf_parse = parser.add_argument_group('CpG Transformer-specific arguments.',
                                        'These arguments are only relevant when imputing with CpG Transformer models.')

deepcpg_parse = parser.add_argument_group('DeepCpG-specific arguments.',
                                        'These arguments are only relevant when imputing with DeepCpG models.')

camelia_parse = parser.add_argument_group('CaMelia-specific arguments.',
                                        'These arguments are only relevant when imputing with CaMelia models.')

# nanmode: return only predictions for positions for which the locally paired similarity feature is defined.


dm_parse.add_argument('--segment_size', type=int, default=1250,
                      help='Bin size in number of CpG sites (columns) that every batch will contain.')
dm_parse.add_argument('--fracs', type=float, nargs='+', default=[1,0,0],
                      help='Fraction of every chromosome that will go to train, val, test respectively. Is ignored for chromosomes that occur in --val_keys or --test_keys.')
dm_parse.add_argument('--mask_p', type=float, default=0.25,
                      help='How many sites to mask each batch as a percentage of the number of columns in the batch.')
dm_parse.add_argument('--mask_random_p', type=float, default=0.20,
                      help='The percentage of masked sites to instead randomize.')
dm_parse.add_argument('--resample_cells', type=int, default=None,
                      help='Whether to resample cells every training batch. Reduces complexity.')
dm_parse.add_argument('--resample_cells_val', type=int, default=None, 
                      help='Whether to resample cells every validation batch.')
dm_parse.add_argument('--val_keys', type=str, nargs='+', default=['chr5'],
                      help='Names/keys of validation chromosomes.')
dm_parse.add_argument('--test_keys', type=str, nargs='+', default=['chr10'], 
                      help='Names/keys of test chromosomes.')
dm_parse.add_argument('--batch_size', type=int, default=1,
                      help='Batch size')
dm_parse.add_argument('--n_workers', type=int, default=4,
                      help='Number of worker threads to use in data loading. Increase if you experience a CPU bottleneck.')