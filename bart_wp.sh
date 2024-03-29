#!/bin/bash0
#SBATCH --job-name  wp
#SBATCH --time      96:00:00
#SBATCH -c          10
#SBATCH --mem       20G
#SBATCH --gpus      1
#SBATCH --mail-type END
#SBATCH --mail-user zzangmane@snu.ac.kr
conda activate torch
ml cuda

python main_bart_embedding_savedir_logdir_isconti_laststep_usemem_usecumul_usegamma_userake_usealpha_usefusion_cumulnum_noibt_nofme_datadir_nepoch_gpuname_batch_istest_debug.py bart_wp bart_wp 0 0 1 1 0 0 1 0 1 0 0 wp_rake 1 cuda:0 8 0 0

python main_bart_embedding_savedir_logdir_isconti_laststep_usemem_usecumul_usegamma_userake_usealpha_usefusion_cumulnum_noibt_nofme_datadir_nepoch_gpuname_batch_istest_debug.py bart_wp bart_wp 0 0 1 1 0 0 1 0 1 0 0 whole 1 cuda:0 1 1 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp/test/generations_outputs_2 completeness-gpt bart_wp_2_completeness-gpt cuda:0 2 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp/test/generations_outputs_5 completeness-gpt bart_wp_5_completeness-gpt cuda:0 5 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp/test/generations_outputs_10 completeness-gpt bart_wp_10_completeness-gpt cuda:0 10 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp/test/generations_outputs_15 completeness-gpt bart_wp_15_completeness-gpt cuda:0 15 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp/test/generations_outputs_20 completeness-gpt bart_wp_20_completeness-gpt cuda:0 20 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp/test/generations_outputs_25 completeness-gpt bart_wp_25_completeness-gpt cuda:0 25 0


python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp/test/generations_outputs_2 nextsentenceprediction-gpt bart_wp_2_nextsentenceprediction-gpt cuda:0 2 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp/test/generations_outputs_5 nextsentenceprediction-gpt bart_wp_5_nextsentenceprediction-gpt cuda:0 5 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp/test/generations_outputs_10 nextsentenceprediction-gpt bart_wp_10_nextsentenceprediction-gpt cuda:0 10 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp/test/generations_outputs_15 nextsentenceprediction-gpt bart_wp_15_nextsentenceprediction-gpt cuda:0 15 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp/test/generations_outputs_20 nextsentenceprediction-gpt bart_wp_20_nextsentenceprediction-gpt cuda:0 20 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp/test/generations_outputs_25 nextsentenceprediction-gpt bart_wp_25_nextsentenceprediction-gpt cuda:0 25 0
