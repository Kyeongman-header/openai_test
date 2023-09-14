#!/bin/bash
#python make_con* valid wprdbk 0 1
#python make_con* train wp 0 1
python make_con* train wp 100000 2
python make_con* train wp 200000 3
python make_con* train rd 0 4
python make_con* train bk 0 5
sh run.sh 
