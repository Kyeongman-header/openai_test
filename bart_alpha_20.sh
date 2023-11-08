#!/bin/bash0
#SBATCH __job_name  alpha20
#SBATCH __time      96:00:00
#SBATCH _c          10
#SBATCH __mem       20G
#SBATCH __gpus      1
#SBATCH __mail_type END
#SBATCH __mail_user zzangmane@snu.ac.kr
conda activate torch
ml cuda

# python main_bart_embedding_savedir_logdir_isconti_laststep_usemem_usecumul_usegamma_userake_usealpha_usefusion_cumulnum_noibt_nofme_datadir_nepoch_gpuname_batch_istest_debug.py bart_alpha20 bart_alpha20 0 0 1 1 0 0 0 0 3 0 0 whole 1 cuda:0 8 0 0 0.2

python main_bart_embedding_savedir_logdir_isconti_laststep_usemem_usecumul_usegamma_userake_usealpha_usefusion_cumulnum_noibt_nofme_datadir_nepoch_gpuname_batch_istest_debug.py bart_alpha20 bart_alpha20 0 0 1 1 0 0 0 0 3 0 0 whole 1 cuda:0 1 1 0 0.2

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_alpha20/test/generations_outputs_5 completeness_gpt bart_alpha20_5_completeness_gpt cuda:0 5 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_alpha20/test/generations_outputs_10 completeness_gpt bart_alpha20_10_completeness_gpt cuda:0 10 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_alpha20/test/generations_outputs_19 completeness_gpt bart_alpha20_19_completeness_gpt cuda:0 19 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_alpha20/test/generations_outputs_30 completeness_gpt bart_alpha20_20_completeness_gpt cuda:0 30 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_alpha20/test/generations_outputs_50 completeness_gpt bart_alpha20_25_completeness_gpt cuda:0 50 0


python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_alpha20/test/generations_outputs_5 nextsentenceprediction_gpt bart_alpha20_5_nextsentenceprediction_gpt cuda:0 5 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_alpha20/test/generations_outputs_10 nextsentenceprediction_gpt bart_alpha20_10_nextsentenceprediction_gpt cuda:0 10 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_alpha20/test/generations_outputs_19 nextsentenceprediction_gpt bart_alpha20_19_nextsentenceprediction_gpt cuda:0 19 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_alpha20/test/generations_outputs_30 nextsentenceprediction_gpt bart_alpha20_30_nextsentenceprediction_gpt cuda:0 30 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_alpha20/test/generations_outputs_50 nextsentenceprediction_gpt bart_alpha20_50_nextsentenceprediction_gpt cuda:0 50 0
