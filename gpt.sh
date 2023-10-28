#!/bin/sh
#SBATCH --job-name  gpt
#SBATCH --time      96:00:00
#SBATCH -c          10
#SBATCH --mem       20G
#SBATCH --gpus      1
#SBATCH --mail-type END
#SBATCH --mail-user zzangmane@snu.ac.kr
source activate torch
#conda activate torch
ml cuda

#python main_gpt_embedding_savedir_logdir_isconti_laststep_usemem_usecumul_usegamma_userake_usealpha_usefusion_cumulnum_noibt_nofme_datadir_nepoch_gpuname_batch_istest_debug.py gpt_rake gpt_rake 0 0 0 0 0 0 0 0 0 0 0 whole 1 cuda:0 4 0 0

python main_gpt_embedding_savedir_logdir_isconti_laststep_usemem_usecumul_usegamma_userake_usealpha_usefusion_cumulnum_noibt_nofme_datadir_nepoch_gpuname_batch_istest_debug.py gpt_rake gpt_rake 0 0 0 0 0 0 0 0 0 0 0 whole 1 cuda:0 1 1 0

python test_longformer_filename_savedir_logdir_gpu_debug.py GPTGenerations/gpt_rake/test/generations_outputs_2 completeness-gpt gpt_2_completeness-gpt cuda:0 2 0

python test_longformer_filename_savedir_logdir_gpu_debug.py GPTGenerations/gpt_rake/test/generations_outputs_5 completeness-gpt gpt_5_completeness-gpt cuda:0 5 0

python test_longformer_filename_savedir_logdir_gpu_debug.py GPTGenerations/gpt_rake/test/generations_outputs_10 completeness-gpt gpt_10_completeness-gpt cuda:0 10 0

python test_longformer_filename_savedir_logdir_gpu_debug.py GPTGenerations/gpt_rake/test/generations_outputs_15 completeness-gpt gpt_15_completeness-gpt cuda:0 15 0

python test_longformer_filename_savedir_logdir_gpu_debug.py GPTGenerations/gpt_rake/test/generations_outputs_20 completeness-gpt gpt_20_completeness-gpt cuda:0 20 0

python test_longformer_filename_savedir_logdir_gpu_debug.py GPTGenerations/gpt_rake/test/generations_outputs_25 completeness-gpt gpt_25_completeness-gpt cuda:0 25 0


python test_longformer_filename_savedir_logdir_gpu_debug.py GPTGenerations/gpt_rake/test/generations_outputs_2 nextsentenceprediction-gpt gpt_2_nextsentenceprediction-gpt cuda:0 2 0

python test_longformer_filename_savedir_logdir_gpu_debug.py GPTGenerations/gpt_rake/test/generations_outputs_5 nextsentenceprediction-gpt gpt_5_nextsentenceprediction-gpt cuda:0 5 0

python test_longformer_filename_savedir_logdir_gpu_debug.py GPTGenerations/gpt_rake/test/generations_outputs_10 nextsentenceprediction-gpt gpt_10_nextsentenceprediction-gpt cuda:0 10 0

python test_longformer_filename_savedir_logdir_gpu_debug.py GPTGenerations/gpt_rake/test/generations_outputs_15 nextsentenceprediction-gpt gpt_15_nextsentenceprediction-gpt cuda:0 15 0

python test_longformer_filename_savedir_logdir_gpu_debug.py GPTGenerations/gpt_rake/test/generations_outputs_20 nextsentenceprediction-gpt gpt_20_nextsentenceprediction-gpt cuda:0 20 0

python test_longformer_filename_savedir_logdir_gpu_debug.py GPTGenerations/gpt_rake/test/generations_outputs_25 nextsentenceprediction-gpt gpt_25_nextsentenceprediction-gpt cuda:0 25 0
