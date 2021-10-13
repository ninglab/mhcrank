#!/bin/bash
outdir=$1

#PBS -V
#PBS -N train_APmodels
#PBS -A PCON0041
#PBS -l nodes=1:ppn=8
#PBS -l walltime=24:00:00
#PBS -m abe
#PBS -M patrick.skillman-lawrence@osumc.edu

##### Main

python $outdir/../../mhc_rank/train_processing_models_command.py --out-models-dir $outdir \
                                                                 --continue-incomplete

