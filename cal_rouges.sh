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

# python cal_rouge.py bartGenerations/bart_rake_lp/test/generations_outputs_5 5
# python cal_rouge.py bartGenerations/bart_rake_lp/test/generations_outputs_19 19
# python cal_rouge.py bartGenerations/bart_rake_lp/test/generations_outputs_30 30
# python cal_rouge.py bartGenerations/bart_rake_lp/test/generations_outputs_50 50

# python cal_rouge.py bartGenerations/bart_rake_nolp/test/generations_outputs_5 5
# python cal_rouge.py bartGenerations/bart_rake_nolp/test/generations_outputs_30 30
# python cal_rouge.py bartGenerations/bart_rake_nolp/test/generations_outputs_50 50

# python cal_rouge.py bartGenerations/bart_rake/test/generations_outputs_19 19
# python cal_rouge.py bartGenerations/bart_rake/test/generations_outputs_30 30
# python cal_rouge.py bartGenerations/bart_rake/test/generations_outputs_50 50

# python cal_rouge.py PlotmachineGenerations/plotmachine_rake/test/generations_outputs_5 5
# python cal_rouge.py PlotmachineGenerations/plotmachine_rake/test/generations_outputs_19 19
# python cal_rouge.py PlotmachineGenerations/plotmachine_rake/test/generations_outputs_30 30
# python cal_rouge.py PlotmachineGenerations/plotmachine_rake/test/generations_outputs_50 50

# python cal_rouge.py GPTGenerations/gpt_nomemnocumul_alpha/test/generations_outputs_5 5
# python cal_rouge.py GPTGenerations/gpt_nomemnocumul_alpha/test/generations_outputs_19 19
# python cal_rouge.py GPTGenerations/gpt_nomemnocumul_alpha/test/generations_outputs_30 30
# python cal_rouge.py GPTGenerations/gpt_nomemnocumul_alpha/test/generations_outputs_50 50

# python cal_rouge.py bartGenerations/bart_nomem_rake/test/generations_outputs_5 5
# python cal_rouge.py bartGenerations/bart_nomem_rake/test/generations_outputs_10 10
# python cal_rouge.py bartGenerations/bart_nomem_rake/test/generations_outputs_19 19
# python cal_rouge.py bartGenerations/bart_nomem_rake/test/generations_outputs_30 30
# python cal_rouge.py bartGenerations/bart_nomem_rake/test/generations_outputs_50 50

# python cal_rouge.py bartGenerations/bart_nocumul_rake/test/generations_outputs_5 5
# python cal_rouge.py bartGenerations/bart_nocumul_rake/test/generations_outputs_19 19
# python cal_rouge.py bartGenerations/bart_nocumul_rake/test/generations_outputs_30 30
# python cal_rouge.py bartGenerations/bart_nocumul_rake/test/generations_outputs_50 50

# python cal_rouge.py bartGenerations/bart_nomemnocumul_rake/test/generations_outputs_5 5
# python cal_rouge.py bartGenerations/bart_nomemnocumul_rake/test/generations_outputs_30 30
# python cal_rouge.py bartGenerations/bart_nomemnocumul_rake/test/generations_outputs_50 50

# python cal_rouge.py bartGenerations/bart_gamma_rake/test/generations_outputs_5 5
# python cal_rouge.py bartGenerations/bart_gamma_rake/test/generations_outputs_10 10
# python cal_rouge.py bartGenerations/bart_gamma_rake/test/generations_outputs_19 19
# python cal_rouge.py bartGenerations/bart_gamma_rake/test/generations_outputs_30 30
# python cal_rouge.py bartGenerations/bart_gamma_rake/test/generations_outputs_50 50

python cal_rouge.py bartGenerations/bart_nofme/test/generations_outputs_5 5
python cal_rouge.py bartGenerations/bart_nofme/test/generations_outputs_10 10
python cal_rouge.py bartGenerations/bart_nofme/test/generations_outputs_19 19
python cal_rouge.py bartGenerations/bart_nofme/test/generations_outputs_30 30
python cal_rouge.py bartGenerations/bart_nofme/test/generations_outputs_50 50