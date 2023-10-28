#!/bin/sh
#SBATCH --job-name  humaneval
#SBATCH --time      96:00:00
#SBATCH -c          10
#SBATCH --mem       30G
#SBATCH --gpus      1
#SBATCH --mail-type END
#SBATCH --mail-user zzangmane@snu.ac.kr
conda activate torch
ml cuda

python make_humaneval_file.py bartGenerations/bart_rake/test/generations_outputs_whole 0 0 10
python make_humaneval_file.py bartGenerations/bart_nomem_rake/test/generations_outputs_whole 0 0 10
python make_humaneval_file.py bartGenerations/bart_nocumul_rake/test/generations_outputs_whole 0 0 10
python make_humaneval_file.py bartGenerations/bart_nomemnocumul_rake/test/generations_outputs_whole 0 0 10
python make_humaneval_file.py bartGenerations/bart_wp_rake/test/generations_outputs_whole 0 0 10
python make_humaneval_file.py bartGenerations/bart_bk_rake/test/generations_outputs_whole 0 0 10
python make_humaneval_file.py bartGenerations/bart_rd_rake/test/generations_outputs_whole 0 0 10
python make_humaneval_file.py bartGenerations/bart_nofme/test/generations_outputs_whole 0 0 10
python make_humaneval_file.py GPTGenerations/gpt_nomemnocumul_alpha/test/generations_outputs_whole 0 0 10
python make_humaneval_file.py PlotmachineGenerations/plotmachine_rake/test/generations_outputs_whole 0 0 10

