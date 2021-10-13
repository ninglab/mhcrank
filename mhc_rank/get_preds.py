import argparse
import pandas as pd
import numpy as np
import json
import sys

from class1_processing_neural_network import Class1ProcessingNeuralNetwork


parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument(
    "--model_name",
    required=True,
    type=str,
    help='Name of the model. Will be used to name the output csv file')
parser.add_argument(
    "--bench",
    required=True,
    type=str,
    help='Name of the benchmarking csv')
parser.add_argument(
    "--dir",
    required=True,
    type=str,
    help="Directory number. For example, if the directroy was ./training_09/, the dir_num would be 09.")
parser.add_argument(
    "--weights",
    required=True,
    type=int,
    help="The number of the wieghts file within the specified training directory")
parser.add_argument(
    "--start",
    type=int,
    default=0,
    help="The index in the benchmarking dataset that the predicitons should start")
parser.add_argument(
    "--chunk",
    type=int,
    default=50000,
    help="The batch size for preditions.")


def get_model(dir_num, weight_num):
    loc = '../training_' + dir_num + '/'
    manifest = pd.read_csv(loc + 'manifest.csv')
    name = manifest.iloc[weight_num, 0]
    weight_file = loc + 'weights_' + name + '.npz'
    weights_npz = np.load(weight_file)
    weights = []
    for weight in weights_npz.files:
        weights.append(weights_npz[weight])
    
    config = json.loads(manifest.iloc[weight_num, 1])
    
    model = Class1ProcessingNeuralNetwork.from_config(config=config, weights=weights)
    
    return model


def main(args):
    #  Define variables
    model = args.model_name
    outfile = model + '.csv'
    bench_df = pd.read_csv(args.bench).loc[:, ['n_flank', 'peptide', 'c_flank']]
    dir_num = args.dir
    weight_num = args.weights
    start = args.start
    batch_size = args.chunk
        
    model = get_model(dir_num, weight_num)
    
    bench_len = bench_df.shape[0]
    size = np.zeros((bench_len,))
    outdf = pd.DataFrame(size, columns=[model])  # create a blank df to store predictions
    for i in range(start, bench_len, batch_size):
        if batch_size + i > bench_len:  # if current idx + batch exceeds len of df,
            end = bench_len             # set end to max length to avoid index error
        else:                           # otherwise, set end to current idx + betch size 
            end = i + batch_size
        size = end - i
        pred_input = bench_df.iloc[i:end, :]  # use i + end to cut bench into chunks to speed up preds
        peptide = pred_input.peptide.tolist()
        n_flank = pred_input.n_flank.tolist()
        c_flank = pred_input.c_flank.tolist()
        pred = model.predict(peptide, n_flank, c_flank)
        pred_out  = np.reshape(pred, (size,1))
        outdf.iloc[i:end] = pred_out  # set pred equal to corresponding idx on outdf
   
    outdf.to_csv(outfile, index=False)  # after all predicitons have been made, save to csv
    

def run(argv=sys.argv[1:]):
    args = parser.parse_args(argv)
    return main(args)


if __name__ == '__main__':
    run()
    
