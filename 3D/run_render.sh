#!/bin/bash -l
#SBATCH -J GS
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32GB
#SBATCH --cpus-per-gpu 4
#SBATCH -p rtx4090
#SBATCH --qos=normal
#SBATCH --output="logs/render-%j.log"
#SBATCH --error="logs/render-%j.err"
#SBATCH --gres=gpu:1

python3 -u render.py --model_path output/lego --eval