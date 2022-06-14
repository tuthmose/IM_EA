#!/bin/tcsh

setenv g16root /usr/local/gaussian/g16b01
source $g16root/g16/bsd/g16.login
setenv GAUSS_EXEDIR $g16root/g16

set inpfile=$1
set outfile=$2
set namedir=$3
set workdir=`pwd`

rm -rf $namedir
mkdir $namedir
cd $namedir

g16 $workdir/$inpfile $workdir/$outfile
cd -
rm -rf $namedir

exit
