#!/bin/bash
#

#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH -J manifolds_dcalexnet_random
#SBATCH --output=/home/annatruzzi/neural_manifolds_replicaMFT/logs/slurm-%j.out
#SBATCH --error=/home/annatruzzi/neural_manifolds_replicaMFT/logs/slurm-%j.err


PYTHON="/opt/anaconda3/envs/manifolds/bin/python3"

CUDA_VISIBLE_DEVICES=0,1 ${PYTHON} manifolds_dcalexnet_random.py