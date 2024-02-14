#!/bin/bash

#SBATCH --job-name=test
#SBATCH --partition=gpu
#SBATCH --time=150:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=3


#SBATCH --mem-per-cpu=20000

###SBATCH --gres=gpu:rtx1080ti:1
###SBATCH --gres=gpu:rtx3090:1
#SBATCH --gres=gpu:1
#SBATCH -o /cluster/work/medinfmk/STCS_swiss_transplant/logs/test.out
#
source ~/.bashrc
enable_modules

module load python/3.8.10 
module load scipy-stack/2022a
source /cluster/work/medinfmk/STCS_swiss_transplant/cec_env/bin/activate
python3 -u /cluster/work/medinfmk/STCS_swiss_transplant/code_ct/pysurvival_mine/train_scripts/cv_2.py