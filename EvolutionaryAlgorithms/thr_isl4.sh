#!/bin/bash
#PBS -l select=1:ncpus=12 -q q07lee -N OronzoCana

module load python/3.6.2

export DATA=/home/g.mancini/bigd/ga_test/
export WORK=/home/g.mancini/bigd/ga_test/Threonine/Island_4
export PYTHONPATH=$DATA/src
export SCRATCH="/local/scratch/g.mancini"
export LOCALDIR="$SCRATCH/$PBS_JOBID"

cd $WORK
rm    -rf $LOCALDIR
mkdir -p $LOCALDIR
cp threonine.tpl $DATA/gaushell.csh $LOCALDIR
cd $LOCALDIR

python3.6 $DATA/conf_GA_parser.py -I threonine.tpl -g 3 1 5 1 7 5 10 5 12 10 -N lthr4_ga_run -l "int=ultrafine opt=(verytight)" \
 -n 40 -C 100 -F 1 -s 0.25 -i 5 -m 0.5 -M 0.2 -p 5 "maxdist" "cosine" |tee  $WORK/l_threonine4.out

rm -f $LOCALDIR/gaushell.csh
cp final_* fitness_* $WORK
cd $WORK
echo `hostname`
echo "All Done"
exit 0
