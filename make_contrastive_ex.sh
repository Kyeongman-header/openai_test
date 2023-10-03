#!/bin/bash
#python make_con* valid wprdbk 0 1
#python make_con* train wp 0 1
python make_contrastive*.py test nextsentenceprediction whole
python make_contrastive*.py train nextsentenceprediction whole
python make_contrastive*.py valid nextsentenceprediction whole
#python make_contrastive*.py test completeness whole
#python make_contrastive*.py train completeness whole
#python make_contrastive*.py valid completeness whole
sh gpt_run.sh
