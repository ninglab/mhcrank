#!/bin/bash
model=$1
bench=$2
dir_num=$3
weight_num=$4
start=$5
chunk=$6

#PBS -V
#PBS -N get_AP_predicitons
#PBS -A PCON0041
#PBS -l nodes=1:ppn=8
#PBS -l walltime=1:00:00
#PBS -m abe
#PBS -M patrick.skillman-lawrence@osumc.edu

python ../../mhc_rank/get_preds.py --model_name $model --bench $bench --dir $dir_num --weights $weight_num --start $start --chunk $chunk
