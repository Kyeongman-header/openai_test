#!/bin/bash0
#SBATCH --job-name  rd
#SBATCH --time      96:00:00
#SBATCH -c          10
#SBATCH --mem       20G
#SBATCH --gpus      1
#SBATCH --mail-type END
#SBATCH --mail-user zzangmane@snu.ac.kr
conda activate torch
ml cuda

python main_bart_embedding_savedir_logdir_isconti_laststep_usemem_usecumul_usegamma_userake_usealpha_usefusion_cumulnum_noibt_nofme_datadir_nepoch_gpuname_batch_istest_debug.py bart_rd bart_rd 0 0 1 1 0 0 1 0 1 0 0 reedsy_rake 1 cuda:0 8 0 0

python main_bart_embedding_savedir_logdir_isconti_laststep_usemem_usecumul_usegamma_userake_usealpha_usefusion_cumulnum_noibt_nofme_datadir_nepoch_gpuname_batch_istest_debug.py bart_rd bart_rd 0 0 1 1 0 0 1 0 1 0 0 whole 1 cuda:0 1 1 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_rd/test/generations_outputs_2 completeness-gpt bart_rd_2_completeness-gpt cuda:0 2 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_rd/test/generations_outputs_5 completeness-gpt bart_rd_5_completeness-gpt cuda:0 5 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_rd/test/generations_outputs_10 completeness-gpt bart_rd_10_completeness-gpt cuda:0 10 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_rd/test/generations_outputs_15 completeness-gpt bart_rd_15_completeness-gpt cuda:0 15 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_rd/test/generations_outputs_20 completeness-gpt bart_rd_20_completeness-gpt cuda:0 20 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_rd/test/generations_outputs_25 completeness-gpt bart_rd_25_completeness-gpt cuda:0 25 0


python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_rd/test/generations_outputs_2 nextsentenceprediction-gpt bart_rd_2_nextsentenceprediction-gpt cuda:0 2 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_rd/test/generations_outputs_5 nextsentenceprediction-gpt bart_rd_5_nextsentenceprediction-gpt cuda:0 5 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_rd/test/generations_outputs_10 nextsentenceprediction-gpt bart_rd_10_nextsentenceprediction-gpt cuda:0 10 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_rd/test/generations_outputs_15 nextsentenceprediction-gpt bart_rd_15_nextsentenceprediction-gpt cuda:0 15 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_rd/test/generations_outputs_20 nextsentenceprediction-gpt bart_rd_20_nextsentenceprediction-gpt cuda:0 20 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_rd/test/generations_outputs_25 nextsentenceprediction-gpt bart_rd_25_nextsentenceprediction-gpt cuda:0 25 0
