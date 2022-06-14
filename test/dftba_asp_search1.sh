#!/bin/bash
#PBS -l select=1:ncpus=08 -q q14curie -N dftba_R1

module load python/3

export ROOT=/home/g.mancini/bigd/ga_test/
export WORK=/home/g.mancini/bigd/ga_test/aminoacids/AsparticAcid/dftba
export BIN=/home/g.mancini/from_bigdata/ga_test/src/
export PYTHONPATH="$PYTHONPATH:$BIN"
export SCRATCH="/local/scratch/g.mancini"
export LOCALDIR="$SCRATCH/$PBS_JOBID"
export TPL="$WORK/dftba_asp_ref_order.tpl"
echo "ENVIRONMENT SET"

cd $WORK
rm -rf $WORK/DFTBA_constr_R1/ dftba_R1.e* dftba_R1.o*
echo $PBS_JOBID 
rm    -rf $LOCALDIR
mkdir -p $LOCALDIR
cp $TPL gaushell.csh $LOCALDIR
cd $LOCALDIR

echo "working area set"

python3 $BIN/conf_GA_parser.py -I $TPL -g 1 4 4 13 4 6 6 9 9 11 13 15 -N asp_dftba_constr_R1 \
 -n 50 -C 100 -s 0.5 -c 0.6 -m 0.5 -M 0.3 -V 4 --cutoff 0.75 \
 -F -l "opt=(verytight,MaxCycles=100,ModRedundant)" -X "rotation3" \
 -i 4 -p 4 "maxdist" "cosine" \
 --hof 0.1 --var 10.\
 1> >(tee $WORK/Asp_dftba_constr_R1.out) 2> >(tee $WORK/Asp_dftba_constr_R1.out)

cp fitness_* $WORK
rm -f $LOCALDIR/gaushell.csh
mkdir Com Log
mv *com Com
mv *log Log
bzip2 Com/*com Log/*log
cd ..
mkdir -p $WORK/DFTBA_constr_R1
rsync -az $LOCALDIR/Com $LOCALDIR/Log $WORK/DFTBA_constr_R1/
cp init_pop* $WORK/DFTBA_constr_R1/
if [ $ret -ne 0 ] ; then
        echo "error copying files, $LOCALDIR on `hostname`"
else
        rm -rf $LOCALDIR
fi
cd $WORK
echo `hostname` 
echo "All Done" 

exit 0
