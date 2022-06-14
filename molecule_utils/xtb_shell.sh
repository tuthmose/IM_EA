#!/bin/bash

source /usr/local/bin/set_xtb.sh

export OMP_NUM_THREADS=$5
if [ "$9" == "--opt" ]; then
    xtb $1 $2 $3 $4 $5 $6 $7 $8 $9 >& ${10}
else
    xtb $1 $2 $3 $4 $5 $6 $7 $8 >& $9
fi
