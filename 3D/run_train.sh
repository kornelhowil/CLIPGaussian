#!/bin/bash -l
#SBATCH -J GS
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32GB
#SBATCH --cpus-per-gpu 4
#SBATCH -p rtx4090
#SBATCH --qos=normal
#SBATCH --output="logs/train-%j.log"
#SBATCH --error="logs/train-%j.err"
#SBATCH --gres=gpu:1

python3 -u train.py -s /shared/results/z1216473/lego --model_path output/lego --ply_path /home/z1216473/gaussian-splatting/output/lego/point_cloud/iteration_30000/point_cloud.ply --iterations 5000 --style_prompt "Starry Night by Vincent van Gogh"