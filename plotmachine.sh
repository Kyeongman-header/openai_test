#!/bin/sh
#SBATCH --job-name  plotmachine
#SBATCH --time      96:00:00
#SBATCH -c          10
#SBATCH --mem       30G
#SBATCH --gpus      1
#SBATCH --mail-type END
#SBATCH --mail-user zzangmane@snu.ac.kr
source activate torch
#conda activate torch
ml cuda

python main_plotmachine_embedding_savedir_logdir_isconti_laststep_usemem_usecumul_usegamma_userake_usealpha_usefusion_cumulnum_noibt_nofme_datadir_nepoch_gpuname_batch_istest_debug.py plotmachine_rake plotmachine_rake 0 0 1 0 0 0 1 0 0 0 1 whole 2 cuda:0 4 0 0

python main_plotmachine_embedding_savedir_logdir_isconti_laststep_usemem_usecumul_usegamma_userake_usealpha_usefusion_cumulnum_noibt_nofme_datadir_nepoch_gpuname_batch_istest_debug.py plotmachine_rake plotmachine_rake 0 0 1 0 0 0 1 0 0 0 1 whole 1 cuda:0 1 1 0

python test_longformer_filename_savedir_logdir_gpu_debug.py PlotmachineGenerations/plotmachine_rake/test/generations_outputs_5 completeness-gpt plotmachine_5_completeness-gpt cuda:0 5 0

python test_longformer_filename_savedir_logdir_gpu_debug.py PlotmachineGenerations/plotmachine_rake/test/generations_outputs_10 completeness-gpt plotmachine_10_completeness-gpt cuda:0 10 0

python test_longformer_filename_savedir_logdir_gpu_debug.py PlotmachineGenerations/plotmachine_rake/test/generations_outputs_19 completeness-gpt plotmachine_19_completeness-gpt cuda:0 19 0

python test_longformer_filename_savedir_logdir_gpu_debug.py PlotmachineGenerations/plotmachine_rake/test/generations_outputs_30 completeness-gpt plotmachine_30_completeness-gpt cuda:0 30 0

python test_longformer_filename_savedir_logdir_gpu_debug.py PlotmachineGenerations/plotmachine_rake/test/generations_outputs_50 completeness-gpt plotmachine_50_completeness-gpt cuda:0 50 0


python test_longformer_filename_savedir_logdir_gpu_debug.py PlotmachineGenerations/plotmachine_rake/test/generations_outputs_5 nextsentenceprediction-gpt plotmachine_5_nextsentenceprediction-gpt cuda:0 5 0

python test_longformer_filename_savedir_logdir_gpu_debug.py PlotmachineGenerations/plotmachine_rake/test/generations_outputs_10 nextsentenceprediction-gpt plotmachine_10_nextsentenceprediction-gpt cuda:0 10 0

python test_longformer_filename_savedir_logdir_gpu_debug.py PlotmachineGenerations/plotmachine_rake/test/generations_outputs_19 nextsentenceprediction-gpt plotmachine_19_nextsentenceprediction-gpt cuda:0 19 0

python test_longformer_filename_savedir_logdir_gpu_debug.py PlotmachineGenerations/plotmachine_rake/test/generations_outputs_30 nextsentenceprediction-gpt plotmachine_30_nextsentenceprediction-gpt cuda:0 30 0

python test_longformer_filename_savedir_logdir_gpu_debug.py PlotmachineGenerations/plotmachine_rake/test/generations_outputs_50 nextsentenceprediction-gpt plotmachine_50_nextsentenceprediction-gpt cuda:0 50 0
