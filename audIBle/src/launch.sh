#!/bin/bash
#SBATCH -c 10
#SBATCH --mem=20G
#SBATCH -p gpu
#SBATCH --gres gpu:rtx8000:1
#SBATCH --time 1-00:00:00
#SBATCH --job-name=vocalset_from_layers
#SBATCH --mail-type=ALL
#SBATCH -o /lium/raid-b/tmario/logs/slurmout/log.%a
#SBATCH -e /lium/raid-b/tmario/logs/slurmout/err.%a
#SBATCH --mail-user=theo.mariotte@univ-lemans.fr

DATASET=$3
python train_ssl_downstream.py --conf_id $1 --seed $2

python eval_classifier_ssl.py --conf_id $1 --seed $2 --exp_tag ssl_downstream_${DATASET} --dataset_name $DATASET --samplerate 16000
