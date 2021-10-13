import argparse
import json
import sys
from math import ceil

parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument(
    "--output",
    required=True,
    type=str,
    help='Directory to save hyperparameters.json')
parser.add_argument(
    "--encoding",
    type=str,
    nargs='+',
    help='List of different amino acid encoding to try.',
    default=['BLOSUM62'])
parser.add_argument(
    "--learn_embedding",
    type=str,
    nargs='+',
    help='Whether the embeddings should be learned (yes/no). combo uses both learned and hardcoded embeddings',
    default=['no'])
parser.add_argument(
    "--learned_embed_dims",
    type=int,
    nargs='+',
    help='The output dimension of learned embedding for each amino acid',
    default=[10])
parser.add_argument(
    "--peptide_max_length",
    type=int,
    nargs='+',
    help='List of lengths to shape peptides to',
    default=[9])
parser.add_argument(
    "--flank_length",
    type=int,
    nargs='+',
    help='List of lengths to shape n and c flank to',
    default=[5])
parser.add_argument(
    "--cleave_radius",
    type=int,
    nargs='+',
    help='Cleaveage radius to use for global kernel',
    default=[3])
parser.add_argument(
    "--conv_n_kernels",
    type=int,
    nargs='+',
    help='Number of distinct kernels/ filters to use',
    default=[16])
parser.add_argument(
    "--conv_kernel_size",
    type=int,
    nargs='+',
    help='Size of kernels to use for convolutions',
    default=[5])
parser.add_argument(
    "--conv_activation",
    type=str,
    nargs='+',
    help='Activation to use for convolution layers',
    default=['relu'])
parser.add_argument(
    "--l1l2",
    type=float,
    nargs='+',
    help='List of lists including L1 L2 learning rates. do not use this arg. Needs to be altered...',
    default=[[0.0, 0.0]])  # do not use this arg. Needs to be altered...
parser.add_argument(
    "--dense_layer_size",
    type=int,
    nargs='+',
    help='Size of dense layers',
    default=[32])
parser.add_argument(
    "--dropout_rate",
    type=float,
    nargs='+',
    help='Dropout rate for dropout layers',
    default=[0.5])
parser.add_argument(
    "--loss_function",
    type=str,
    nargs='+',
    help='Loss functions to try',
    default=["binary_crossentropy"])
parser.add_argument(
    "--optimizer",
    type=str,
    nargs='+',
    help='Optimizers to try',
    default=['adam'])
parser.add_argument(
    "--combos_per_json",
    default=0,
    type=int,
    help="How many architectures should be in each file. Default of 0 indicates "
    "all the architectures should be saved to a single file")


def main(args):

    hyperparameters = []
    for encoding in args.encoding:
        for pep_length in args.peptide_max_length:
            assert pep_length >= 8
            assert pep_length <= 15
            for f_length in args.flank_length:
                assert f_length >= 0
                assert f_length <= 5
                for radii in args.cleave_radius:
                    assert radii <= f_length
                    for nfilter in args.conv_n_kernels:
                        for kernel_size in args.conv_kernel_size:
                            for activation in args.conv_activation:
                                for learn in args.l1l2:
                                    for dense_size in args.dense_layer_size:
                                        for dropout in args.dropout_rate:
                                            for loss in args.loss_function:
                                                for optimize in args.optimizer:
                                                    for opt in args.learn_embedding:
                                                        for dim in args.learned_embed_dims:
                                                            architecture = dict()
                                                            architecture['amino_acid_encoding'] = encoding
                                                            architecture['learn_embedding'] = opt
                                                            architecture['learned_embed_dims'] = dim
                                                            architecture['peptide_max_length'] = pep_length
                                                            architecture['n_flank_length'] = f_length
                                                            architecture['c_flank_length'] = f_length
                                                            architecture['cleave_radius'] = radii
                                                            architecture['convolutional_filters'] = nfilter
                                                            architecture['convolutional_kernel_size'] = kernel_size
                                                            architecture['convolutional_activation'] = activation
                                                            architecture['convolutional_kernel_l1_l2'] = learn
                                                            architecture['post_convolutional_dense_layer_sizes'] = dense_size
                                                            architecture['dropout_rate'] = dropout
                                                            architecture['loss_function'] = loss
                                                            architecture['optimizer'] = optimize
                                                            hyperparameters.append(architecture)
   
    output = args.output
    n_archs = args.combos_per_json
    
    if n_archs == 0:
        file = output + 'hyperparameters.json' 
        with open(file, 'w') as out_file:
            json.dump(hyperparameters, out_file)
    else:
        total_archs = len(hyperparameters)
        n_files = ceil(total_archs / n_archs)
        for i in range(n_files):
            file = output + 'hyperparameters_{:02d}.json'.format(i)
            arch = i * n_archs
            if i == n_files - 1:
                lst = hyperparameters[arch:]
            else:
                lst = hyperparameters[arch:arch+n_archs]
            with open(file, 'w') as out_file:
                json.dump(lst, out_file)
        

def run(argv=sys.argv[1:]):
    args = parser.parse_args(argv)
    return main(args)


if __name__ == '__main__':
    run()
