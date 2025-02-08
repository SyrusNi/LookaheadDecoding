#!/bin/bash
#SBATCH -o ./experiments/%j.out
#SBATCH -e ./experiments/%j.err
#SBATCH -p a800
#SBATCH --qos=normal
#SBATCH -J lookahead
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
#SBATCH --mem 40GB

mkdir ./experiments/$SLURM_JOB_ID

export CUDA=0
export LADE=1
export LEVEL=5
export WIN=15
export GUESS=15
export FLASH=0
export PP=0
CUDA_VISIBLE_DEVICES=$CUDA USE_LADE=$LADE python applications/eval_mtbench.py \
    --model-path models/vicuna-7b-v1.3 \
    --model-id vicuna-7b-v1.3-level-$LEVEL-win-$WIN-guess-$GUESS-f$FLASH-pp$CUDA \
    --level $LEVEL --window $WIN --guess $GUESS --use-flash $FLASH --use-pp $PP