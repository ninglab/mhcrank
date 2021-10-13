#!/bin/bash
csv=$1
folds=$2
hyperparameters=$3
outdir=$4

#PBS -V
#PBS -N init_APmodels
#PBS -A PCON0041
#PBS -l nodes=1:ppn=8
#PBS -l walltime=01:00:00
#PBS -m abe
#PBS -M patrick.skillman-lawrence@osumc.edu

##### Main

python $outdir/../../mhc_rank/train_processing_models_command.py --data $csv \
                                                                 --hyperparameters $hyperparameters \
                                                                 --out-models-dir $outdir \
                                                                 --pre-folded $folds \
                                                                 --only-initialize

