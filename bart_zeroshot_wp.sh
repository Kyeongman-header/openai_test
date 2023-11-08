#!/bin/bash0
#SBATCH __job_name  wp
#SBATCH __time      96:00:00
#SBATCH _c          10
#SBATCH __mem       20G
#SBATCH __gpus      1
#SBATCH __mail_type END
#SBATCH __mail_user zzangmane@snu.ac.kr
conda activate torch
ml cuda

# python main_bart_embedding_savedir_logdir_isconti_laststep_usemem_usecumul_usegamma_userake_usealpha_usefusion_cumulnum_noibt_nofme_datadir_nepoch_gpuname_batch_istest_debug.py bart_wp bart_wp 0 0 1 1 0 0 1 0 1 0 0 wp_rake 1 cuda:0 8 0 0

python main_bart_embedding_savedir_logdir_isconti_laststep_usemem_usecumul_usegamma_userake_usealpha_usefusion_cumulnum_noibt_nofme_datadir_nepoch_gpuname_batch_istest_debug.py bart_wp_rake bart_wp_zero_bk 0 0 1 1 0 0 1 0 3 0 0 booksum_rake 1 cuda:0 1 1 0

python main_bart_embedding_savedir_logdir_isconti_laststep_usemem_usecumul_usegamma_userake_usealpha_usefusion_cumulnum_noibt_nofme_datadir_nepoch_gpuname_batch_istest_debug.py bart_wp_rake bart_wp_zero_rd 0 0 1 1 0 0 1 0 3 0 0 reedsy_rake 1 cuda:0 1 1 0


python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp_zero_bk/test/generations_outputs_5 completeness_gpt bart_wp_zero_bk_5_completeness_gpt cuda:0 5 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp_zero_bk/test/generations_outputs_10 completeness_gpt bart_wp_zero_bk_10_completeness_gpt cuda:0 10 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp_zero_bk/test/generations_outputs_19 completeness_gpt bart_wp_zero_bk_19_completeness_gpt cuda:0 19 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp_zero_bk/test/generations_outputs_30 completeness_gpt bart_wp_zero_bk_30_completeness_gpt cuda:0 30 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp_zero_bk/test/generations_outputs_50 completeness_gpt bart_wp_zero_bk_50_completeness_gpt cuda:0 50 0


python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp_zero_bk/test/generations_outputs_5 nextsentenceprediction_gpt bart_wp_zero_bk_5_nextsentenceprediction_gpt cuda:0 5 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp_zero_bk/test/generations_outputs_10 nextsentenceprediction_gpt bart_wp_zero_bk_10_nextsentenceprediction_gpt cuda:0 10 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp_zero_bk/test/generations_outputs_19 nextsentenceprediction_gpt bart_wp_zero_bk_19_nextsentenceprediction_gpt cuda:0 19 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp_zero_bk/test/generations_outputs_30 nextsentenceprediction_gpt bart_wp_zero_bk_30_nextsentenceprediction_gpt cuda:0 30 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp_zero_bk/test/generations_outputs_50 nextsentenceprediction_gpt bart_wp_zero_bk_50_nextsentenceprediction_gpt cuda:0 50 0



python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp_zero_rd/test/generations_outputs_5 completeness_gpt bart_wp_zero_rd_5_completeness_gpt cuda:0 5 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp_zero_rd/test/generations_outputs_10 completeness_gpt bart_wp_zero_rd_10_completeness_gpt cuda:0 10 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp_zero_rd/test/generations_outputs_19 completeness_gpt bart_wp_zero_rd_19_completeness_gpt cuda:0 19 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp_zero_rd/test/generations_outputs_30 completeness_gpt bart_wp_zero_rd_30_completeness_gpt cuda:0 30 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp_zero_rd/test/generations_outputs_50 completeness_gpt bart_wp_zero_rd_50_completeness_gpt cuda:0 50 0


python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp_zero_rd/test/generations_outputs_5 nextsentenceprediction_gpt bart_wp_zero_rd_5_nextsentenceprediction_gpt cuda:0 5 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp_zero_rd/test/generations_outputs_10 nextsentenceprediction_gpt bart_wp_zero_rd_10_nextsentenceprediction_gpt cuda:0 10 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp_zero_rd/test/generations_outputs_19 nextsentenceprediction_gpt bart_wp_zero_rd_19_nextsentenceprediction_gpt cuda:0 19 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp_zero_rd/test/generations_outputs_30 nextsentenceprediction_gpt bart_wp_zero_rd_30_nextsentenceprediction_gpt cuda:0 30 0

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp_zero_rd/test/generations_outputs_50 nextsentenceprediction_gpt bart_wp_zero_rd_50_nextsentenceprediction_gpt cuda:0 50 0