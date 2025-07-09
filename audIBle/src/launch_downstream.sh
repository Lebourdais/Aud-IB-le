#!/bin/sh
#SBATCH --array=15-28%2
#SBATCH -c 10
#SBATCH --mem=20G
#SBATCH -p gpu
#SBATCH --gres gpu:rtx8000:1
#SBATCH --time 2-00:00:00
#SBATCH --job-name=timit_from_layers
#SBATCH --mail-type=ALL
#SBATCH -o /lium/raid-b/tmario/logs/slurmout/log.%a
#SBATCH -e /lium/raid-b/tmario/logs/slurmout/err.%a
#SBATCH --mail-user=theo.mariotte@univ-lemans.fr

f="$(sed "${SLURM_ARRAY_TASK_ID}q;d" ./exp_list.lst)"
echo $f >> out.log
./launch.sh $f
