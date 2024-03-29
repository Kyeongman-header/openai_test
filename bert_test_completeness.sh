#!/bin/sh
#SBATCH --job-name  bert-completeness
#SBATCH --time      96:00:00
#SBATCH -c          10
#SBATCH --mem       30G
#SBATCH --gpus      1
#SBATCH --mail-type END
#SBATCH --mail-user zzangmane@snu.ac.kr
source activate torch
#conda activate torch
ml cuda
# python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_rake/test/generations_outputs_1 bert-completeness bart_rake_1_bert-completeness cuda:0 0
# sleep 1s
# python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_rake/test/generations_outputs_2 bert-completeness bart_rake_2_bert-completeness cuda:0 0
# sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_rake_lp/test/generations_outputs_5 bert-completeness bart_rake_lp_5_bert-completeness cuda:0 5 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_rake_lp/test/generations_outputs_10 bert-completeness bart_rake_lp_10_bert-completeness cuda:0 10 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_rake_lp/test/generations_outputs_19 bert-completeness bart_rake_lp_19_bert-completeness cuda:0 19 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_rake_lp/test/generations_outputs_30 bert-completeness bart_rake_lp_30_bert-completeness cuda:0 30 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_rake_lp/test/generations_outputs_50 bert-completeness bart_rake_lp_50_bert-completeness cuda:0 50 0
sleep 1s
# python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nomem_rake/test/generations_outputs_1 bert-completeness bart_nomem_rake_1_bert-completeness cuda:0 0
# sleep 1s
# python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nomem_rake/test/generations_outputs_2 bert-completeness bart_nomem_rake_2_bert-completeness cuda:0 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nomem_rake/test/generations_outputs_5 bert-completeness bart_nomem_rake_5_bert-completeness cuda:0 5 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nomem_rake/test/generations_outputs_10 bert-completeness bart_nomem_rake_10_bert-completeness cuda:0 10 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nomem_rake/test/generations_outputs_19 bert-completeness bart_nomem_rake_19_bert-completeness cuda:0 19 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nomem_rake/test/generations_outputs_30 bert-completeness bart_nomem_rake_30_bert-completeness cuda:0 30 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nomem_rake/test/generations_outputs_50 bert-completeness bart_nomem_rake_50_bert-completeness cuda:0 50 0
sleep 1s

# python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nocumul_rake/test/generations_outputs_1 bert-completeness bart_nocumul_rake_1_bert-completeness cuda:0 0
# sleep 1s
# python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nocumul_rake/test/generations_outputs_2 bert-completeness bart_nocumul_rake_2_bert-completeness cuda:0 0
# sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nocumul_rake/test/generations_outputs_5 bert-completeness bart_nocumul_rake_5_bert-completeness cuda:0 5 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nocumul_rake/test/generations_outputs_10 bert-completeness bart_nocumul_rake_10_bert-completeness cuda:0 10 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nocumul_rake/test/generations_outputs_19 bert-completeness bart_nocumul_rake_19_bert-completeness cuda:0 19 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nocumul_rake/test/generations_outputs_30 bert-completeness bart_nocumul_rake_30_bert-completeness cuda:0 30 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nocumul_rake/test/generations_outputs_50 bert-completeness bart_nocumul_rake_50_bert-completeness cuda:0 50 0
sleep 1s

# python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nomemnocumul_rake/test/generations_outputs_1 bert-completeness bart_nomemnocumul_rake_1_bert-completeness cuda:0 0
# sleep 1s
# python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nomemnocumul_rake/test/generations_outputs_2 bert-completeness bart_nomemnocumul_rake_2_bert-completeness cuda:0 0
# sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nomemnocumul_rake/test/generations_outputs_5 bert-completeness bart_nomemnocumul_rake_5_bert-completeness cuda:0 5 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nomemnocumul_rake/test/generations_outputs_10 bert-completeness bart_nomemnocumul_rake_10_bert-completeness cuda:0 10 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nomemnocumul_rake/test/generations_outputs_19 bert-completeness bart_nomemnocumul_rake_19_bert-completeness cuda:0 19 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nomemnocumul_rake/test/generations_outputs_30 bert-completeness bart_nomemnocumul_rake_30_bert-completeness cuda:0 30 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nomemnocumul_rake/test/generations_outputs_50 bert-completeness bart_nomemnocumul_rake_50_bert-completeness cuda:0 50 0
sleep 1s
#python bert_test_longformer_filename_savedir_logdir_gpu_debug.py PlotmachineGenerations/plotmachine_rake/test/generations_outputs_1 bert-completeness plotmachine_rake_1_bert-completeness cuda:0 0
#python bert_test_longformer_filename_savedir_logdir_gpu_debug.py PlotmachineGenerations/plotmachine_rake/test/generations_outputs_2 bert-completeness plotmachine_rake_2_bert-completeness cuda:0 0
# sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py PlotmachineGenerations/plotmachine_rake/test/generations_outputs_5 bert-completeness plotmachine_rake_5_bert-completeness cuda:0 5 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py PlotmachineGenerations/plotmachine_rake/test/generations_outputs_10 bert-completeness plotmachine_rake_10_bert-completeness cuda:0 10 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py PlotmachineGenerations/plotmachine_rake/test/generations_outputs_19 bert-completeness plotmachine_rake_19_bert-completeness cuda:0 19 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py PlotmachineGenerations/plotmachine_rake/test/generations_outputs_30 bert-completeness plotmachine_rake_30_bert-completeness cuda:0 30 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py PlotmachineGenerations/plotmachine_rake/test/generations_outputs_50 bert-completeness plotmachine_rake_50_bert-completeness cuda:0 50 0
sleep 1s

python bert_test_longformer_filename_savedir_logdir_gpu_debug.py GPTGenerations/gpt_nomemnocumul_alpha/test/generations_outputs_5  bert-completeness gpt_5_bert-completeness cuda:0 5 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py GPTGenerations/gpt_nomemnocumul_alpha/test/generations_outputs_10 bert-completeness gpt_10_bert-completeness cuda:0 10 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py GPTGenerations/gpt_nomemnocumul_alpha/test/generations_outputs_19 bert-completeness gpt_19_bert-completeness cuda:0 19 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py GPTGenerations/gpt_nomemnocumul_alpha/test/generations_outputs_30 bert-completeness gpt_30_bert-completeness cuda:0 30 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py GPTGenerations/gpt_nomemnocumul_alpha/test/generations_outputs_50 bert-completeness gpt_50_bert-completeness cuda:0 50 0
sleep 1s

python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp_rake/test/generations_outputs_5 bert-completeness bart_wp_rake_5_bert-completeness cuda:0 5 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp_rake/test/generations_outputs_10 bert-completeness bart_wp_rake_10_bert-completeness cuda:0 10 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp_rake/test/generations_outputs_19 bert-completeness bart_wp_rake_19_bert-completeness cuda:0 19 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp_rake/test/generations_outputs_30 bert-completeness bart_wp_rake_30_bert-completeness cuda:0 30 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp_rake/test/generations_outputs_50 bert-completeness bart_wp_rake_50_bert-completeness cuda:0 50 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nofme/test/generations_outputs_5 bert-completeness bart_nofme_5_bert-completeness cuda:0 5 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nofme/test/generations_outputs_10 bert-completeness bart_nofme_10_bert-completeness cuda:0 10 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nofme/test/generations_outputs_19 bert-completeness bart_nofme_19_bert-completeness cuda:0 19 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nofme/test/generations_outputs_30 bert-completeness bart_nofme_30_bert-completeness cuda:0 30 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nofme/test/generations_outputs_50 bert-completeness bart_nofme_50_bert-completeness cuda:0 50 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_bk_rake/test/generations_outputs_5 bert-completeness bart_bk_rake_5_bert-completeness cuda:0 5 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_bk_rake/test/generations_outputs_10 bert-completeness bart_bk_rake_10_bert-completeness cuda:0 10 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_bk_rake/test/generations_outputs_19 bert-completeness bart_bk_rake_19_bert-completeness cuda:0 19 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_bk_rake/test/generations_outputs_30 bert-completeness bart_bk_rake_30_bert-completeness cuda:0 30 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_bk_rake/test/generations_outputs_50 bert-completeness bart_bk_rake_50_bert-completeness cuda:0 50 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_rd_rake/test/generations_outputs_5 bert-completeness bart_rd_rake_5_bert-completeness cuda:0 5 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_rd_rake/test/generations_outputs_10 bert-completeness bart_rd_rake_10_bert-completeness cuda:0 10 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_rd_rake/test/generations_outputs_19 bert-completeness bart_rd_rake_19_bert-completeness cuda:0 19 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_rd_rake/test/generations_outputs_30 bert-completeness bart_rd_rake_30_bert-completeness cuda:0 30 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_rd_rake/test/generations_outputs_50 bert-completeness bart_rd_rake_50_bert-completeness cuda:0 50 0

python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_gamma_rake/test/generations_outputs_5 bert-completeness bart_gamma_rake_5_bert-completeness cuda:0 5 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_gamma_rake/test/generations_outputs_10 bert-completeness bart_gamma_rake_10_bert-completeness cuda:0 10 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_gamma_rake/test/generations_outputs_19 bert-completeness bart_gamma_rake_19_bert-completeness cuda:0 19 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_gamma_rake/test/generations_outputs_30 bert-completeness bart_gamma_rake_30_bert-completeness cuda:0 30 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_gamma_rake/test/generations_outputs_50 bert-completeness bart_gamma_rake_50_bert-completeness cuda:0 50 0

python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_rake_nolp/test/generations_outputs_5 bert-completeness bart_rake_nolp_5_bert-completeness cuda:0 5 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_rake_nolp/test/generations_outputs_10 bert-completeness bart_rake_nolp_10_bert-completeness cuda:0 10 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_rake_nolp/test/generations_outputs_19 bert-completeness bart_rake_nolp_19_bert-completeness cuda:0 19 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_rake_nolp/test/generations_outputs_30 bert-completeness bart_rake_nolp_30_bert-completeness cuda:0 30 0
sleep 1s
python bert_test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_rake_nolp/test/generations_outputs_50 bert-completeness bart_rake_nolp_50_bert-completeness cuda:0 50 0
