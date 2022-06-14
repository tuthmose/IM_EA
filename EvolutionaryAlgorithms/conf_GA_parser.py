## G Mancini Dec 18

import argparse as argp
import math
import numpy as np    
import os.path     
import re
import subprocess as sub 
#
from sys  import argv, exit  

# custom modules
import ga_evolution, ga_population, ga_utils, gen_init_pop
import zmatrix
import gau_parser
import xtb_parser

Parse = argp.ArgumentParser(description='Search conformers with gaussian and GAs')
# template and gaussian settings
Parse.add_argument('-A','--always',help='do not block run if gaussian has an error',default=False,action='store_true')
Parse.add_argument('-b','--check_bonds',help='forbid gaussian to broke or create new bonds',default=False,action='store_true')
Parse.add_argument('-B','--bdelta',help='threshold for bond breaking',default=0.2,type=float)
Parse.add_argument('-e','--notconv',help='bogus energy value if not converged',default=100.0,type=float)
Parse.add_argument('-E','--stopok',help='wheter to consider converged a stopped opt.',default=False,action='store_true')
Parse.add_argument('-g','--genes',help='rotatable bonds to be rotated in GA',default=None,type=int,nargs='+')
Parse.add_argument('-I','--INP',help='template gaussian input file (zmatrix)',default='template.com',type=str)
Parse.add_argument('-N','--NAME',help='file name prefix for gaussian files',default='TMP',type=str)
Parse.add_argument('-R','--RES',help='dihedral resolution',default=30.,type=float)
Parse.add_argument('-S','--SHELL',help='shell script to run ext. prog.',default="gaushell.csh",type=str)
Parse.add_argument('-V','--verbose',help='verbosity level (0 to 5)',default=2,type=int)
Parse.add_argument('-v','--var',help='variance for dihedral sampling',default=10.,type=float)
Parse.add_argument('-x','--cutoff',help='cutoff (angstrom) for clash check',default=0.90,type=float)
Parse.add_argument('-K','--kwargs',help='arguments for fitcalc',default=None,type=str)
### xTB instead of gaussian
Parse.add_argument('--xTB',help='use xTB instead of G16: template xtb file',default=None, type=str)
### GA SETTINGS
Parse.add_argument('-F','--Flex',help='Optimize geometry (Lamarckian GA)',default=False,action='store_true')
Parse.add_argument('-l','--g16_opt',help='g16 optimization options',default=None,type=str)
#Parse.add_argument('-l','--g16_opt',help='g16 optimization options',\
#default='opt(verytight) int(ultrafine) MaxCycle=50',type=str)
Parse.add_argument('-L','--LS',help="do Linear Search of chromosomes after the last iteration",\
   default=False,action='store_true')
Parse.add_argument('-f','--fhigh',help='best fitness is lowest (default) or highest',\
   default=False,action='store_true')
Parse.add_argument('-n','--niter',help='number of iterations',default=10,type=int)
Parse.add_argument('-C','--nchrm',help='number of chromosomes',default=20,type=int)
Parse.add_argument('-c','--pCo',help='cross over probability',default=0.5,type=float)
Parse.add_argument('-M','--ppmut',help='parents mutation probability',default=0.2,type=float)
Parse.add_argument('-m','--pcmut',help='children mutation probability',default=0.4,type=float)
Parse.add_argument('-k','--co_meth',help='cross over method (default rotation)',default="rotation",type=str)
Parse.add_argument('-s','--sel_press',help='selection pressure',default=0.5,type=float)
Parse.add_argument('-r','--rank',help='rank parent selection for last iterations',default=True,action='store_true')
Parse.add_argument('-t','--tsize',help='tournament size',default=2,type=int)
Parse.add_argument('-i','--isles',help='use island model with n islands',default=1,type=int)
Parse.add_argument('-p','--migration',help='migration frequency and policy, distance',default=None,nargs=3)
Parse.add_argument('-X','--mutme',help='mutation method (default rotation)',default="rotation",type=str)
Parse.add_argument('-D','--debug',help='print debug messages',default=False,action='store_true')
Parse.add_argument('--seed',help='random seed',action='store',type=int)
Parse.add_argument('--restart',help='do not generate random population: read df with coords and energies',\
                   default=None,action='store',type=str)
Parse.add_argument('--force_ls',help='force linear search after EA',default=False,action='store_true')
Parse.add_argument('--dry_run',help='do not call ESC, just generate structures',default=False,action='store_true')
Parse.add_argument('--no_l_h',help='create init pop with latin hypercube',default=False,\
    action='store_true')
Parse.add_argument('--hof',help='hall of fame size',default=0.0,type=float)
Parse.add_argument('--distrib',help='distribution for dihedral space',default='normal',action='store')
Myarg = Parse.parse_args()
print(Myarg)

mytuple = list()
for i in range(0,len(Myarg.genes),2):
    mytuple.append((Myarg.genes[i],Myarg.genes[i+1]))
Myarg.genes = tuple(mytuple)
Myarg.ngenes = len(Myarg.genes)

#given in form kwd:value kwd:value
KWDS = dict()
KWDS['flex'] = False
if Myarg.kwargs is not None:
    mykwargs = Myarg.kwargs.split()
    for i in mykwargs:
        kv = i.split(":")
        KWDS[kv[0]] = kv[1]
if Myarg.Flex:
    KWDS['flex'] = True

try:
    os.path.isfile(Myarg.INP)
except False:
    raise FileNotFoundError("ERROR: Template zmatrix not found")

if Myarg.migration == None:
    Myarg.migration = tuple((Myarg.niter+1,"round_robin","cosine")) 


###------- SETUP
dihedral_space = ga_utils.dihedral_space(Myarg.RES,Myarg.var)

#set caller initialization
dry_run = not Myarg.dry_run
if Myarg.xTB:
    mycaller = xtb_parser.xTBcaller(template=Myarg.xTB, filename=Myarg.NAME, notconv=Myarg.notconv,\
        stopok=Myarg.stopok, run_esc=dry_run, sptype="mol")
else:
    mycaller = gau_parser.gaucaller(template=Myarg.INP, filename=Myarg.NAME, shellname=Myarg.SHELL,\
    notconv=Myarg.notconv, stopok=Myarg.stopok, optstr=Myarg.g16_opt, \
    run_alw=Myarg.always, cutoff=Myarg.cutoff, check_bonds=Myarg.check_bonds, bdelta=Myarg.bdelta,\
    run_esc=dry_run)

if Myarg.restart == None:
    mypop = gen_init_pop.init_pop_dihedral(Myarg.nchrm, Myarg.INP, Myarg.genes, Myarg.cutoff, \
        Myarg.verbose, dihedral_space, Myarg.notconv, Myarg.xTB, Myarg.no_l_h, mycaller)
else:
        try:
            data = np.loadtxt(Myarg.restart)
            assert data.shape[0] == Myarg.nchrm
            assert data.shape[1] == len(Myarg.genes) + 1
            chrm = data[:,:-1]
            E    = data[:,-1]
            Zmatlist = list()
            for i in range(Myarg.nchrm):
                zmat = ga_utils.linear_molecule(Myarg.INP,Myarg.genes,alleles=chrm[i],cutoff=Myarg.cutoff)
                zmat.set_fitness(E[i])
                Zmatlist.append(zmat)
            mypop = ga_population.GA_population(chromosomes=chrm,fitness=E,\
            specimens=Zmatlist,template=Myarg.INP,genes=Myarg.genes)
            if Myarg.verbose > 4:
                print("generated specimen ", mypop.chromosomes,mypop.fitness)
            np.savetxt("init_pop.dat",mypop.chromosomes)
            
        except:
            raise ValueError("Error loading restart file")
            

###------------ EVOLUTION

print("+++ RUNNING GA with settings:")
print(Myarg) 
if Myarg.Flex:
    print("WARNING: doing a geometry optmization for each fitness evaluation")
    
myGA = ga_evolution.GenAlg(last_rank=Myarg.rank, sel_press=Myarg.sel_press, pCo=Myarg.pCo,\
           ppmut=Myarg.ppmut, pcmut=Myarg.pcmut, verbose=Myarg.verbose, co_meth=Myarg.co_meth,\
           fhigh=Myarg.fhigh, tsize=Myarg.tsize, seed=Myarg.seed,mut=Myarg.mutme,\
           cutoff=Myarg.cutoff,debug=Myarg.debug,hof=Myarg.hof)

fitness, bestc, bestID = myGA.Evolve(genotype=dihedral_space,pop=mypop, niter=Myarg.niter,\
            ffunc=mycaller.fitcalc, fitkwds=KWDS, LS=Myarg.LS,nisle=Myarg.isles,\
            migr_freq=int(Myarg.migration[0]),\
            mpolicy=Myarg.migration[1], distance=Myarg.migration[2], force_ls=Myarg.force_ls)
    
###---------- CLOSE

# print fitness
outf = open("fitness_" + Myarg.NAME + ".dat","w")
for i in range(Myarg.niter):
    outf.write("{0:5d} {1:15.9f} {2:5d}".format(i,fitness[i],bestID[i]))
    for j in range(Myarg.ngenes):
        outf.write("  {0:9.6f}  ".format(bestc[i][j]))
    outf.write("\n")

quit()
