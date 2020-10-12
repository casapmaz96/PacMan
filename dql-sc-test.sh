#! /bin/bash


#SBATCH --job-name="PACMAN"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --output job%J.output
#SBATCH --error job%J.err
#SBATCH --gres=gpu:1
#SBATCH --partition=normal



module load cuda/10.0
module add python/intel

python3 pacman.py -p DQLAgent -x 0 -n 20 -l smallClassic -t -q
