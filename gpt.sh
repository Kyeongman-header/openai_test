#!/bin/sh
#SBATCH --job-name  gpt
#SBATCH --time      96:00:00
#SBATCH -c          10
#SBATCH --mem       30G
#SBATCH --gpus      1
#SBATCH --mail-type END
#SBATCH --mail-user zzangmane@snu.ac.kr
source activate torch
#conda activate torch
ml cuda

python main_gpt_embedding_savedir_logdir_isconti_laststep_usemem_usecumul_usegamma_userake_usealpha_usefusion_cumulnum_noibt_nofme_datadir_nepoch_gpuname_batch_istest_debug.py gpt_rake gpt_rake 0 0 0 0 0 0 0 0 0 0 0 whole 2 cuda:0 4 0 0

python main_gpt_embedding_savedir_logdir_isconti_laststep_usemem_usecumul_usegamma_userake_usealpha_usefusion_cumulnum_noibt_nofme_datadir_nepoch_gpuname_batch_istest_debug.py gpt_rake gpt_rake 0 0 0 0 0 0 0 0 0 0 0 whole 1 cuda:0 1 1 0

python test_longformer_filename_savedir_logdir_gpu_debug.py GPTGenerations/gpt_rake/test/generations_outputs_5 completeness-gpt gpt_5_completeness-gpt cuda:0 5 0

python test_longformer_filename_savedir_logdir_gpu_debug.py GPTGenerations/gpt_rake/test/generations_outputs_10 completeness-gpt gpt_10_completeness-gpt cuda:0 10 0

python test_longformer_filename_savedir_logdir_gpu_debug.py GPTGenerations/gpt_rake/test/generations_outputs_19 completeness-gpt gpt_19_completeness-gpt cuda:0 19 0

python test_longformer_filename_savedir_logdir_gpu_debug.py GPTGenerations/gpt_rake/test/generations_outputs_30 completeness-gpt gpt_30_completeness-gpt cuda:0 30 0

python test_longformer_filename_savedir_logdir_gpu_debug.py GPTGenerations/gpt_rake/test/generations_outputs_50 completeness-gpt gpt_50_completeness-gpt cuda:0 50 0


python test_longformer_filename_savedir_logdir_gpu_debug.py GPTGenerations/gpt_rake/test/generations_outputs_5 nextsentenceprediction-gpt gpt_5_nextsentenceprediction-gpt cuda:0 5 0

python test_longformer_filename_savedir_logdir_gpu_debug.py GPTGenerations/gpt_rake/test/generations_outputs_10 nextsentenceprediction-gpt gpt_10_nextsentenceprediction-gpt cuda:0 10 0

python test_longformer_filename_savedir_logdir_gpu_debug.py GPTGenerations/gpt_rake/test/generations_outputs_19 nextsentenceprediction-gpt gpt_19_nextsentenceprediction-gpt cuda:0 19 0

python test_longformer_filename_savedir_logdir_gpu_debug.py GPTGenerations/gpt_rake/test/generations_outputs_30 nextsentenceprediction-gpt gpt_30_nextsentenceprediction-gpt cuda:0 30 0

python test_longformer_filename_savedir_logdir_gpu_debug.py GPTGenerations/gpt_rake/test/generations_outputs_50 nextsentenceprediction-gpt gpt_50_nextsentenceprediction-gpt cuda:0 50 0
