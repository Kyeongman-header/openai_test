#!/bin/sh
#SBATCH --job-name  plotmachine
#SBATCH --time      96:00:00
#SBATCH -c          10
#SBATCH --mem       20G
#SBATCH --gpus      1
#SBATCH --mail-type END
#SBATCH --mail-user zzangmane@snu.ac.kr
source activate torch
#conda activate torch
ml cuda

python main_plotmachine_embedding_savedir_logdir_isconti_laststep_usemem_usecumul_usegamma_userake_usealpha_usefusion_cumulnum_noibt_nofme_datadir_nepoch_gpuname_batch_istest_debug.py plotmachine_rake plotmachine_rake 0 0 1 0 0 0 0 0 0 0 1 whole 1 cuda:0 4 0 0 

python main_plotmachine_embedding_savedir_logdir_isconti_laststep_usemem_usecumul_usegamma_userake_usealpha_usefusion_cumulnum_noibt_nofme_datadir_nepoch_gpuname_batch_istest_debug.py plotmachine_rake plotmachine_rake 0 0 1 0 0 0 0 0 0 0 1 whole 1 cuda:0 1 1 0

python test_longformer_filename_savedir_logdir_gpu_debug.py PlotmachineGenerations/plotmachine_rake/test/generations_outputs_2 completeness-gpt plotmachine_2_completeness-gpt cuda:0 2 0

python test_longformer_filename_savedir_logdir_gpu_debug.py PlotmachineGenerations/plotmachine_rake/test/generations_outputs_5 completeness-gpt plotmachine_5_completeness-gpt cuda:0 5 0

python test_longformer_filename_savedir_logdir_gpu_debug.py PlotmachineGenerations/plotmachine_rake/test/generations_outputs_10 completeness-gpt plotmachine_10_completeness-gpt cuda:0 10 0

python test_longformer_filename_savedir_logdir_gpu_debug.py PlotmachineGenerations/plotmachine_rake/test/generations_outputs_15 completeness-gpt plotmachine_15_completeness-gpt cuda:0 15 0

python test_longformer_filename_savedir_logdir_gpu_debug.py PlotmachineGenerations/plotmachine_rake/test/generations_outputs_20 completeness-gpt plotmachine_20_completeness-gpt cuda:0 20 0

python test_longformer_filename_savedir_logdir_gpu_debug.py PlotmachineGenerations/plotmachine_rake/test/generations_outputs_25 completeness-gpt plotmachine_25_completeness-gpt cuda:0 25 0


python test_longformer_filename_savedir_logdir_gpu_debug.py PlotmachineGenerations/plotmachine_rake/test/generations_outputs_2 nextsentenceprediction-gpt plotmachine_2_nextsentenceprediction-gpt cuda:0 2 0

python test_longformer_filename_savedir_logdir_gpu_debug.py PlotmachineGenerations/plotmachine_rake/test/generations_outputs_5 nextsentenceprediction-gpt plotmachine_5_nextsentenceprediction-gpt cuda:0 5 0

python test_longformer_filename_savedir_logdir_gpu_debug.py PlotmachineGenerations/plotmachine_rake/test/generations_outputs_10 nextsentenceprediction-gpt plotmachine_10_nextsentenceprediction-gpt cuda:0 10 0

python test_longformer_filename_savedir_logdir_gpu_debug.py PlotmachineGenerations/plotmachine_rake/test/generations_outputs_15 nextsentenceprediction-gpt plotmachine_15_nextsentenceprediction-gpt cuda:0 15 0

python test_longformer_filename_savedir_logdir_gpu_debug.py PlotmachineGenerations/plotmachine_rake/test/generations_outputs_20 nextsentenceprediction-gpt plotmachine_20_nextsentenceprediction-gpt cuda:0 20 0

python test_longformer_filename_savedir_logdir_gpu_debug.py PlotmachineGenerations/plotmachine_rake/test/generations_outputs_25 nextsentenceprediction-gpt plotmachine_25_nextsentenceprediction-gpt cuda:0 25 0
