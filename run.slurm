#! /bin/bash
# ============================
#SBATCH --job-name=ENEL645-Assignement2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=0-10:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
# ==================================

eval "$(~/miniforge3/bin/conda shell.bash hook)"

conda init
conda activate sam_env

python BuildModel.py