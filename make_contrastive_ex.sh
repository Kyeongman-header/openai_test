#!/bin/bash
#python make_con* valid wprdbk 0 1
#python make_con* train wp 0 1
python bert_make_contrastive*.py test nextsentenceprediction_gpt whole
python bert_make_contrastive*.py train nextsentenceprediction_gpt whole
python bert_make_contrastive*.py valid nextsentenceprediction_gpt whole
#python bert_make_contrastive*.py test completeness_gpt whole
#python bert_make_contrastive*.py train completeness_gpt whole
#python bert_make_contrastive*.py valid completeness_gpt whole
sh gpt_run.sh
