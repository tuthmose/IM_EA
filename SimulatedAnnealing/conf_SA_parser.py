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
sys.path.insert(0, '/home/gmancini/Dropbox/appunti/SearchAlgos/EvolutionaryAlgorithms/src/')

# custom modules
import simulated_annealing, sa_utils
import ga_utils
import zmatrix
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
Parse.add_argument('-r','--rot_bonds',help='rotatable bonds to be rotated in SA',default=None,type=int,nargs='+')
Parse.add_argument('-I','--INP',help='template gaussian input file',default='template.com',type=str)
Parse.add_argument('-N','--NAME',help='file name prefix for gaussian files',default='TMP',type=str)
Parse.add_argument('-R','--RES',help='dihedral resolution',default=30.,type=float)
Parse.add_argument('-S','--SHELL',help='shell script to run ext. prog.',default="gaushell.csh",type=str)
Parse.add_argument('-v','--var',help='variance for dihedral sampling',default=10.,type=float)
Parse.add_argument('-x','--cutoff',help='cutoff (angstrom) for clash check',default=0.6,type=float)
Parse.add_argument('-O','--okwargs',help='keyword arguments for obj. func.',default=None,type=str)
Parse.add_argument('-F','--Flex',help='Optimize geometry',default=False,type=bool)
### SA SETTINGS
Parse.add_argument('-b','--bias',help='plain MC: go back to worst conf sampled after n iterations',\
    default=0,type=int)
Parse.add_argument('-l','--g16_opt',help='g16 optimization options',default='',type=str)
Parse.add_argument('-C','--ckwargs',help='keyword arguments for cooling scheme',default=None,type=str)
Parse.add_argument('-g','--gkwargs',help='keyword arguments for gen. func.',default=None,type=str)
Parse.add_argument('-G','--genfunc',help='function used to generate new conformers \
   (provided in imported modules)',default='sa_utils.gen_mol')
Parse.add_argument('-f','--lower_fail',help='lower temp also on rejected moves (default=True)',\
    default=True,action='store_false')
Parse.add_argument('-n','--niter',help='number of iterations',default=10,type=int)
Parse.add_argument('-p','--glob_prob',help='probability to generate a new mol',default=0.,type=float)
Parse.add_argument('-T','--tstart',help='initial temperature',default=1000,type=float)
Parse.add_argument('-c','--cooling_rate',help='cooling rate',default=0.995,type=float)
Parse.add_argument('-B','--boltzmann',help='boltzmann',default=KB,type=float)
Parse.add_argument('-s','--cooling_scheme',help='cooling scheme (provided in imported module)',\
    default='simulated_annealing.linear_cooling')
Parse.add_argument('-V','--verbose',help='verbosity level of SA',default=0,type=int)
Parse.add_argument('-Z','--zm_in',help='use zmat instead of cart. coord. to generate input files',\
                   default=False,action='store_true')
Myarg = Parse.parse_args()

#
dihedral_space = ga_utils.dihedral_space(Myarg.RES,Myarg.var)

assert len(Myarg.rot_bonds) % 2 == 0
nangles = len(Myarg.rot_bonds) // 2
mytuple = list()
for i in range(0,len(Myarg.rot_bonds),2):
    mytuple.append((Myarg.rot_bonds[i],Myarg.rot_bonds[i+1]))
Myarg.rot_bonds = tuple(mytuple)

try:
    os.path.isfile(Myarg.INP)
except False:
    raise FileNotFoundError("ERROR: Template zmatrix not found")

if Myarg.rot_bonds == None:
    raise ValueError("Provide atom pairs for rotatable bonds")
if Myarg.genfunc == None:
    raise ValueError("Provide function to generate new configurations")
if Myarg.cooling_scheme == None:
    raise ValueError("Provide cooling_scheme")

#given in form kwd:value kwd:value
KWDS = [None, None, None]
for i, ikwargs in enumerate((Myarg.okwargs, Myarg.ckwargs, Myarg.gkwargs)):
    if ikwargs is not None:
        mykwargs = ikwargs.split()
        KWDS[i] = dict()
        for j,J in enumerate(mykwargs):
            kv = i.split(":")
            KWDS[i][kv[0]] = kv[1]

# add necessary information to genfunc
if KWDS[2] ==  None:
    KWDS[2] = dict()
KWDS[2]['domain'] = dihedral_space
KWDS[2]['cutoff'] = Myarg.var 
KWDS[2]['glob_prob'] = Myarg.glob_prob

###------- SETUP
conf0 = sa_utils.molecule(Myarg.INP,Myarg.rot_bonds)
    
#gaucaller initialization
mycaller = gau_parser.gaucaller(template=Myarg.INP,filename=Myarg.NAME,shellname=Myarg.SHELL,\
    notconv=Myarg.notconv,stopok=Myarg.stopok,Lamarck=Myarg.Flex,optstr=Myarg.g16_opt,zm_in=Myarg.zm_in)
init_dih,E0 = mycaller.fitcalc(conf0,flex=False)
print("Starting from ",init_dih,E0)

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
            conf0=conf0, domain=dihedral_space, cooling_rate=Myarg.cooling_rate,\
            lower_fail=Myarg.lower_fail, cutoff=Myarg.var*2.)

bestc, Energy, final_temp, nacc = mySA.Anneal(verbose=bool(Myarg.verbose),\
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
