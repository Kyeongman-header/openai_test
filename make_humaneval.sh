#!/bin/sh

conda activate torch

python make_humaneval_file.py bartGenerations/bart_rake/generations_outputs_whole 0 0 30
python make_humaneval_file.py bartGenerations/bart_nomem_rake/generations_outputs_whole 0 0 30
python make_humaneval_file.py bartGenerations/bart_nocumul_rake/generations_outputs_whole 0 0 30
python make_humaneval_file.py bartGenerations/bart_nomemnocumul_rake/generations_outputs_whole 0 0 30
python make_humaneval_file.py bartGenerations/bart_wp_rake/generations_outputs_whole 0 0 30
python make_humaneval_file.py bartGenerations/bart_bk_rake/generations_outputs_whole 0 0 30
python make_humaneval_file.py bartGenerations/bart_rd_rake/generations_outputs_whole 0 0 30
python make_humaneval_file.py bartGenerations/bart_nofme/generations_outputs_whole 0 0 30
python make_humaneval_file.py GPTGenerations/gpt_nomemnocumul_alpha/generations_outputs_whole 0 0 30
python make_humaneval_file.py PlotmachineGenerations/plotmachine_rake/generations_outputs_whole 0 0 30

