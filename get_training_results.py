""" IMPORT PACKAGES """
import sys
from os import getcwd
from glob import glob
import pandas as pd
import numpy as np
import argparse
import json

"""Constants """
hyperparams = ['learn_embedding', 'learned_embed_dims', 'peptide_length',
               'cleave_radius', 'n_kernels', 'kernel_size', 'dense_layer_size']


"""Arg Parser"""
parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument(
    "--num_archs",
    type=int,
    metavar="n_folds",
    default=4,
    required=True,
    help=""
    )
parser.add_argument(
    "--num_folds",
    type=int,
    metavar="n_folds",
    default=4,
    required=True,
    help=""
    )
parser.add_argument(
    "--out",
    type=str,
    metavar="out.csv",
    required=True,
    help=""
)
    

"""Functions """

def main(args):
    n_folds = args.num_folds
    archs = args.num_archs
    dir = args.out
    
    cols = hyperparams.copy()
    for i in range(n_folds):
        cols.append('fold_'+str(i)+'_train_auc')
        cols.append('fold_'+str(i)+'_test_auc')
    for i in range(n_folds):
        cols.append('fold_'+str(i)+'_weights_file')
    
    zeros = np.zeros((archs, len(cols)))
    outdf = pd.DataFrame(zeros, columns=cols)    
    
    pth = getcwd()
    df = pd.read_csv(dir+'/manifest.csv')
    for j in range(df.shape[0]):
        data = json.loads(df.iloc[j, 1])
        ### These would need to be changed if other hyperparameters adjusted
        learn_embedding = data['hyperparameters']['learn_embedding']
        learned_embed_dims = data['hyperparameters']['learned_embed_dims']
        peptide_length = data['hyperparameters']['peptide_max_length']
        cleave_radius = data['hyperparameters']['cleave_radius']
        n_kernels = data['hyperparameters']['convolutional_filters']
        kernel_size = data['hyperparameters']['convolutional_kernel_size']
        dense_layer_size = data['hyperparameters']['post_convolutional_dense_layer_sizes']
        
        fold = data['fit_info'][0]['training_info']['fold_num']
        train_auc = data['fit_info'][0]['training_info']['train_auc']
        test_auc = data['fit_info'][0]['training_info']['test_auc']
        
        file = glob('./' + dir + "/weights_CLEAVAGE-CLASSI-" + str(j) + "-*.npz")[0]
        weights = pth + file[1:]
        
        row = [learn_embedding, learned_embed_dims, peptide_length, cleave_radius,
                n_kernels, kernel_size, dense_layer_size]
        
        try:
            idx = outdf[(outdf['learn_embedding'] == learn_embedding) &
                        (outdf['learned_embed_dims'] == learned_embed_dims) &
                        (outdf['peptide_length'] == peptide_length) &
                        (outdf['cleave_radius'] == cleave_radius) &
                        (outdf['n_kernels'] == n_kernels) &
                        (outdf['kernel_size'] == kernel_size) &
                        (outdf['dense_layer_size'] == dense_layer_size)
                    ].index.tolist()[0]
        except:
            idx = outdf[outdf['learn_embedding'] == 0].index.tolist()[0]
            
        outdf.iloc[idx, :7] = row
        outdf.loc[idx, 'fold_'+str(fold)+'_train_auc'] = train_auc
        outdf.loc[idx, 'fold_'+str(fold)+'_test_auc'] = test_auc
        outdf.loc[idx, 'fold_'+str(fold)+'_weights_file'] = weights
        outdf['mean_test_auc'] = outdf[['fold_0_test_auc','fold_1_test_auc',
                                        'fold_2_test_auc','fold_3_test_auc']].mean(axis=1)
        outdf['std_test_auc'] = outdf[['fold_0_test_auc','fold_1_test_auc',
                                           'fold_2_test_auc','fold_3_test_auc']].std(axis=1)
            
    outdf.to_csv(dir+'/training_results.csv', index=False)


def run(argv=sys.argv[1:]):
    args = parser.parse_args(argv)
    return main(args)


if __name__ == '__main__':
    run()
