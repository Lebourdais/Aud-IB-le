#!/bin/bash
#SBATCH -N1
#SBATCH -c 8
#SBATCH --gres gpu:rtx6000:1
#SBATCH --mem=15G
#SBATCH -J audio_sae
#SBATCH -p gpu
#SBATCH --time 5-00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=theo.mariotte@univ-lemans.fr
python ../../audIBle/src/pretrain_autoencoder.py --conf_id $1 --seed $2
