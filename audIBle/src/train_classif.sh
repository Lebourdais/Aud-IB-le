#!/bin/bash
#SBATCH -N1
#SBATCH -c 8
#SBATCH --gres gpu:rtx8000:1
#SBATCH --mem=20G
#SBATCH -J classif_sae
#SBATCH -p gpu
#SBATCH --time 0-10:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=theo.mariotte@univ-lemans.fr

python train_classifier_asae_urbansound8k.py --conf_id $1 --seed $2
