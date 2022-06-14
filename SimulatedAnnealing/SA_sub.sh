#!/bin/bash

export WORK=/home/gmancini/devel/ML/SA_test
export PYTHONPATH=$WORK:/home/gmancini/devel/ML/src_ga_17jan/

cd $WORK

rm -f sa_test_run_* 
python3 $WORK/conf_SA_parser.py -I fast.tpl -r 1 2 1 5 5 6 -N sa_test_run_near -F 1 -n 100 -T 500 -c 0.99 -l 'opt=loose' \
 -s 'simulated_annealing.velocity_cooling' -f 1 -p 1. >& $WORK/log_near

#python3.6 $WORK/conf_SA_parser.py -I template.tpl -r 1 2 1 5 5 6 -N sa_test_run_global -n 200 -T 1000 >& $WORK/log_global

exit 0
