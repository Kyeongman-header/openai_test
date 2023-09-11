#!/bin/sh

python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_rake/test/generations_outputs_1  nextsentenceprediction-whole bart_rake_1_nextsentenceprediction-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_rake/test/generations_outputs_2  nextsentenceprediction-whole bart_rake_2_nextsentenceprediction-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_rake/test/generations_outputs_5  nextsentenceprediction-whole bart_rake_5_nextsentenceprediction-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_rake/test/generations_outputs_10 nextsentenceprediction-whole bart_rake_10_nextsentenceprediction-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_rake/test/generations_outputs_19 nextsentenceprediction-whole bart_rake_19_nextsentenceprediction-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_rake/test/generations_outputs_30 nextsentenceprediction-whole bart_rake_30_nextsentenceprediction-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_rake/test/generations_outputs_50 nextsentenceprediction-whole bart_rake_50_nextsentenceprediction-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nomem_rake/test/generations_outputs_1  nextsentenceprediction-whole bart_nomem_rake_1_nextsentenceprediction-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nomem_rake/test/generations_outputs_2  nextsentenceprediction-whole bart_nomem_rake_2_nextsentenceprediction-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nomem_rake/test/generations_outputs_5  nextsentenceprediction-whole bart_nomem_rake_5_nextsentenceprediction-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nomem_rake/test/generations_outputs_10 nextsentenceprediction-whole bart_nomem_rake_10_nextsentenceprediction-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nomem_rake/test/generations_outputs_19 nextsentenceprediction-whole bart_nomem_rake_19_nextsentenceprediction-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nocumul_rake/test/generations_outputs_1  nextsentenceprediction-whole bart_nocumul_rake_1_nextsentenceprediction-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nocumul_rake/test/generations_outputs_2  nextsentenceprediction-whole bart_nocumul_rake_2_nextsentenceprediction-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nocumul_rake/test/generations_outputs_5  nextsentenceprediction-whole bart_nocumul_rake_5_nextsentenceprediction-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nocumul_rake/test/generations_outputs_10 nextsentenceprediction-whole bart_nocumul_rake_10_nextsentenceprediction-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nocumul_rake/test/generations_outputs_19 nextsentenceprediction-whole bart_nocumul_rake_19_nextsentenceprediction-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nomemnocumul_rake/test/generations_outputs_1  nextsentenceprediction-whole bart_nomemnocumul_rake_1_nextsentenceprediction-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nomemnocumul_rake/test/generations_outputs_2  nextsentenceprediction-whole bart_nomemnocumul_rake_2_nextsentenceprediction-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nomemnocumul_rake/test/generations_outputs_5  nextsentenceprediction-whole bart_nomemnocumul_rake_5_nextsentenceprediction-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nomemnocumul_rake/test/generations_outputs_10 nextsentenceprediction-whole bart_nomemnocumul_rake_10_nextsentenceprediction-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nomemnocumul_rake/test/generations_outputs_19 nextsentenceprediction-whole bart_nomemnocumul_rake_19_nextsentenceprediction-whole cuda:0 0

#python test_longformer_filename_savedir_logdir_gpu_debug.py PlotmachineGenerations/plotmachine_rake/test/generations_outputs_1  nextsentenceprediction-whole plotmachine_rake_1_nextsentenceprediction-whole cuda:0 0
#python test_longformer_filename_savedir_logdir_gpu_debug.py PlotmachineGenerations/plotmachine_rake/test/generations_outputs_2  nextsentenceprediction-whole plotmachine_rake_2_nextsentenceprediction-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py PlotmachineGenerations/plotmachine_rake/test/generations_outputs_5  nextsentenceprediction-whole plotmachine_rake_5_nextsentenceprediction-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py PlotmachineGenerations/plotmachine_rake/test/generations_outputs_10 nextsentenceprediction-whole plotmachine_rake_10_nextsentenceprediction-whole cuda:0 0
#python test_longformer_filename_savedir_logdir_gpu_debug.py PlotmachineGenerations/plotmachine_rake/test/generations_outputs_19 nextsentenceprediction-whole plotmachine_rake_19_nextsentenceprediction-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp_rake/test/generations_outputs_5 nextsentenceprediction-whole bart_wp_rake_5_nextsentence-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp_rake/test/generations_outputs_10 nextsenteceprediction-whole bart_wp_rake_10_nextsentence-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp_rake/test/generations_outputs_19 nextsenteceprediction-whole bart_wp_rake_19_nextsentence-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp_rake/test/generations_outputs_30 nextsenteceprediction-whole bart_wp_rake_30_nextsentence-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_wp_rake/test/generations_outputs_50 nextsenteceprediction-whole bart_wp_rake_50_nextsentence-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nofme/test/generations_outputs_5 nextsentenceprediction-whole bart_nofme_5_nextsentence-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nofme/test/generations_outputs_10 nextsentenceprediction-whole bart_nofme_10_nextsentence-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nofme/test/generations_outputs_19 nextsentenceprediction-whole bart_nofme_19_nextsentence-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nofme/test/generations_outputs_30 nextsentenceprediction-whole bart_nofme_30_nextsentence-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_nofme/test/generations_outputs_50 nextsentenceprediction-whole bart_nofme_50_nextsentence-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_bk_rake/test/generations_outputs_5 nextsentenceprediction-whole bart_bk_rake_5_nextsentenceprediction-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_bk_rake/test/generations_outputs_10 nextsentenceprediction-whole bart_bk_rake_10_nextsentenceprediction-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_bk_rake/test/generations_outputs_19 nextsentenceprediction-whole bart_bk_rake_19_nextsentenceprediction-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_bk_rake/test/generations_outputs_30 nextsentenceprediction-whole bart_bk_rake_30_nextsentenceprediction-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_bk_rake/test/generations_outputs_50 nextsentenceprediction-whole bart_bk_rake_50_nextsentenceprediction-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_rd_rake/test/generations_outputs_5 nextsentenceprediction-whole bart_rd_rake_5_nextsentenceprediction-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_rd_rake/test/generations_outputs_10 nextsentenceprediction-whole bart_rd_rake_10_nextsentenceprediction-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_rd_rake/test/generations_outputs_19 nextsentenceprediction-whole bart_rd_rake_19_nextsentenceprediction-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_rd_rake/test/generations_outputs_30 nextsentenceprediction-whole bart_rd_rake_30_nextsentenceprediction-whole cuda:0 0
sleep 1s
python test_longformer_filename_savedir_logdir_gpu_debug.py bartGenerations/bart_rd_rake/test/generations_outputs_50 nextsentenceprediction-whole bart_rd_rake_50_nextsentenceprediction-whole cuda:0 0