#!/bin/bash
#SBATCH --job-name  cal_rouge
#SBATCH --time      96:00:00
#SBATCH -c          10
#SBATCH --mem       20G
#SBATCH --gpus      1
#SBATCH --mail-type END
#SBATCH --mail-user zzangmane@snu.ac.kr
conda activate torch
ml cuda

python cal_rouge.py bartGenerations/bart_alpha10/test/generations_outputs_2 2
python cal_rouge.py bartGenerations/bart_alpha10/test/generations_outputs_10 10
python cal_rouge.py bartGenerations/bart_gamma_rake/test/generations_outputs_2 2
python cal_rouge.py bartGenerations/bart_nomemnocheat_rake/test/generations_outputs_2 2
python cal_rouge.py bartGenerations/bart_nomem_rake/test/generations_outputs_2 2
python cal_rouge.py bartGenerations/bart_nocheat_rake/test/generations_outputs_2 2
python cal_rouge.py PlotmachineGenerations/plotmachine_rake/test/generations_outputs_2 2