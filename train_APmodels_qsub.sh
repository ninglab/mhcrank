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
if [[ -n $SLURM_JOB_ID ]] ; then
    SCRIPT_DIR=$(realpath $(dirname $(scontrol show job $SLURM_JOB_ID | awk -F= '/Command=/{print $2}' | cut -d" " -f1)))
else
    SCRIPT_DIR=$(dirname $(realpath $0))
fi
python $SCRIPT_DIR/mhc_rank/train_processing_models_command.py --out-models-dir $outdir \
                                                               --continue-incomplete

