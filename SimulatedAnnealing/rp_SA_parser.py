## G Mancini Dec 18

import argparse as argp
import math
import numpy as np    
import os.path     
import re
from scipy import constants
import subprocess as sub 
#
import sys
from sys  import argv, exit  
#sys.path.insert(0, '/home/gmancini/Dropbox/appunti/SearchAlgos/EvolutionaryAlgorithms/src/')

# custom modules
import simulated_annealing, sa_utils, sa_reacpair
import gau_parser

# boltzmann constant in hartree / K
KB = constants.k
hartree = constants.physical_constants["Hartree energy"][0]
KB = KB / hartree

Parse = argp.ArgumentParser(description='Search conformers with gaussian and SA')
# template and gaussian settings
Parse.add_argument('-a','--delete',help='delete temporary files at the end',default=False,type=bool)
Parse.add_argument('-e','--notconv',help='bogus energy value if not converged',default=100.0,type=float)
Parse.add_argument('-E','--stopok',help='wheter to consider converged a stopped opt.',default=False,type=bool)
Parse.add_argument('-I','--INP',help='template gaussian input file',default='template.tpl',type=str)
Parse.add_argument('-N','--NAME',help='file name prefix for gaussian files',default='TMP',type=str)
Parse.add_argument('-S','--SHELL',help='shell script to run ext. prog.',default="gaushell.csh",type=str)
Parse.add_argument('-x','--cutoff',help='cutoff (angstrom) for clash check',default=0.6,type=float)
Parse.add_argument('-F','--Flex',help='Optimize geometry',default=False,type=bool)
Parse.add_argument('-f','--fragments',help='fragment definition',type=int,nargs=4)
Parse.add_argument('-p','--pivots',help='List of pivot atoms in AB and CD',type=int,nargs="+")
Parse.add_argument('-r','--run_esc',help='run esc or dry run',action='store_false')
Parse.add_argument('-O','--okwargs',help='keyword arguments for obj. func.',default=None,type=str)
### SA SETTINGS
Parse.add_argument('-b','--bias',help='plain MC: go back to worst conf sampled after n iterations',\
    default=0,type=int)
Parse.add_argument('-l','--g16_opt',help='g16 optimization options',default='',type=str)
Parse.add_argument('-L','--lower_fail',help='lower temp also on rejected moves (default=True)',\
    default=True,action='store_false')
Parse.add_argument('-G','--genfunc',help='function used to generate new conformers \
   (provided in imported modules)')
Parse.add_argument('-g','--gkwargs',help='keyword arguments for gen. func.',default=None,type=str)
Parse.add_argument('-n','--niter',help='number of iterations',default=10,type=int)
Parse.add_argument('-T','--tstart',help='initial temperature',default=1000,type=float)
Parse.add_argument('-c','--cooling_rate',help='cooling rate',default=0.995,type=float)
Parse.add_argument('-s','--cooling_scheme',help='cooling scheme (provided in imported module)',\
    default='simulated_annealing.linear_cooling')
Parse.add_argument('--seed',help='random seed',default=None,type=int)
Parse.add_argument('-C','--ckwargs',help='keyword arguments for cooling scheme',default=None,type=str)
Parse.add_argument('-B','--boltzmann',help='boltzmann',default=KB,type=float)
Parse.add_argument('-V','--verbose',help='verbosity level of SA',default=0,type=int)
Myarg = Parse.parse_args()

try:
    os.path.isfile(Myarg.INP)
except False:
    raise FileNotFoundError("ERROR: Template zmatrix not found")

if Myarg.genfunc == None:
    raise ValueError("Provide function to generate new configurations")
if Myarg.cooling_scheme == None:
    raise ValueError("Provide cooling a scheme")

#given in form kwd:value kwd:value
KWDS = [None, None, None]
for i, ikwargs in enumerate((Myarg.okwargs, Myarg.ckwargs, Myarg.gkwargs)):
    if ikwargs is not None:
        mykwargs = ikwargs.split()
        KWDS[i] = dict()
        for i,I in enumerate(mykwargs):
            kv = I.split(":")
            KWDS[i][kv[0]] = kv[1]

if Myarg.seed != None:
    np.random.seed(Myarg.seed)


###------- SETUP
fragments = tuple(((Myarg.fragments[0],Myarg.fragments[1]),(Myarg.fragments[2],Myarg.fragments[3])))
conf0 = sa_reacpair.reac_pair(Myarg.INP,AB=fragments[0],CD=fragments[1],ijkl=Myarg.pivots)
   
#gaucaller initialization
mycaller = gau_parser.gaucaller(template=Myarg.INP,filename=Myarg.NAME,shellname=Myarg.SHELL,\
    notconv=Myarg.notconv,stopok=Myarg.stopok,Lamarck=Myarg.Flex,optstr=Myarg.g16_opt, \
    sptype="clust",run_esc=Myarg.run_esc)

#genfunc and colling scheme initialization
genfunc = simulated_annealing.get_func_from_name(Myarg.genfunc)
cooling_scheme = simulated_annealing.get_func_from_name(Myarg.cooling_scheme)

###-------- RUN SA
print("+++ START SIMULATED ANNEALING")
print("+++ scale factor (boltzmann constant) = {0:g}".format(Myarg.boltzmann))

if Myarg.Flex:
    print("WARNING: doing a geometry optmization for each fitness evaluation")
    
mySA = simulated_annealing.SimulatedAnnealing(objfunc=mycaller.fitcalc, okwds=KWDS[0],\
            cooling_scheme=cooling_scheme, ckwds=KWDS[1], genfunc=genfunc, gkwds=KWDS[2],\
            conf0=conf0, domain=None, cooling_rate=Myarg.cooling_rate,\
            lower_fail=Myarg.lower_fail,cutoff=None)

bestc, Energy, final_temp, nacc = mySA.Anneal(verbose=Myarg.verbose,\
    tstart=Myarg.tstart,nsteps=Myarg.niter, Boltzmann=Myarg.boltzmann, restart=Myarg.bias)

print("+++ \n Best conf={0:d}\n Final Energy={1:f}\n acc_ratio={3:f}\n Final temperature={2:f}\n+++".\
    format(int(bestc), Energy, final_temp, nacc))
    
###---------- CLOSE

# print energy
np.savetxt("sa_energy_" + Myarg.NAME + ".dat",mySA.Energies)

if Myarg.delete:
    args = 'rm -f ./'+Myarg.NAME+'*'
    sub.run(args,stderr=sub.PIPE,shell=True)

quit()
