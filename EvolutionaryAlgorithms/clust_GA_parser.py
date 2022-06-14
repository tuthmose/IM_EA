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
import ga_evolution, ga_population, ga_utils, cluster_utils, gen_init_pop
import zmatrix
import gau_parser
import xtb_parser

Parse = argp.ArgumentParser(description='Search conformers with gaussian and GAs')
# template and gaussian settings
Parse.add_argument('-A','--always',help='do not block run if gaussian has an error',default=False,action='store_true')
Parse.add_argument('-a','--delete',help='delete temporary files at the end',default=False,type=bool)
Parse.add_argument('-e','--notconv',help='bogus energy value if not converged',default=100.0,type=float)
Parse.add_argument('-E','--stopok',help='wheter to consider converged a stopped opt.',\
    default=False,action='store_true')
Parse.add_argument('-g','--genes',help='atoms to be moved in GA (default: all)',\
    default=None,type=int,nargs='+')
Parse.add_argument('-I','--INP',help='template gaussian input file',default='template.com',type=str)
Parse.add_argument('-N','--NAME',help='file name prefix for gaussian files',default='TMP',type=str)
Parse.add_argument('-S','--SHELL',help='shell script to run ext. prog.',default="gaushell.csh",type=str)
Parse.add_argument('-V','--verbose',help='verbosity level (0 to 5)',default=2,type=int)
Parse.add_argument('-v','--var',help='variance for dihedral sampling',default=10.,type=float)
Parse.add_argument('-d','--displ',help='displacement mean and sigma for random moves',\
    default=(0.1,0.05),nargs=2,type=float)
Parse.add_argument('-x','--cutoff',help='cutoff (angstrom) for clash check',default=0.7,type=float)
Parse.add_argument('-K','--kwargs',help='arguments for fitcalc',default=None,type=str)
### xTB instead of gaussian
Parse.add_argument('--xTB',help='use xTB instead of G16',default=False,action='store_true')
### GA SETTINGS
Parse.add_argument('-F','--Flex',help='Optimize geometry (Lamarckian GA)',default=False,action='store_true')
Parse.add_argument('-l','--g16_opt',help='g16 optimization options',default='',type=str)
#Parse.add_argument('-l','--g16_opt',help='g16 optimization options',\
#default='opt(verytight) int(ultrafine) MaxCycle=50',type=str)
Parse.add_argument('-L','--LS',help="do Linear Search of chromosomes after the last iteration",default=False,type=bool)
Parse.add_argument('-f','--fhigh',help='best fitness is lowest (default) or highest',default=0,type=bool)
Parse.add_argument('-n','--niter',help='number of iterations',default=10,type=int)
Parse.add_argument('-C','--nchrm',help='number of chromosomes',default=20,type=int)
Parse.add_argument('-c','--pCo',help='cross over probability',default=0.5,type=float)
Parse.add_argument('-M','--ppmut',help='parents mutation probability',default=0.2,type=float)
Parse.add_argument('-m','--pcmut',help='children mutation probability',default=0.4,type=float)
Parse.add_argument('-k','--co_meth',help='cross over method',default="custom",type=str)
Parse.add_argument('-s','--sel_press',help='selection pressure',default=0.5,type=float)
Parse.add_argument('-r','--rank',help='rank parent selection for last iterations',default=True,type=bool)
Parse.add_argument('-t','--tsize',help='tournament size',default=2,type=int)
Parse.add_argument('-u','--upd',help='seed update frequency (default 0)',default=0,type=int)
Parse.add_argument('-i','--isles',help='use island model with n islands',default=1,type=int)
Parse.add_argument('-p','--migration',help='migration frequency and policy, distance',default=None,nargs=3)
Parse.add_argument('-X','--mut_meth',help='mutation method',default="custom",type=str)
Parse.add_argument('-D','--debug',help='print debug messages',default=False,action='store_true')
Parse.add_argument('--dry_run',help='do not call ESC, just generate structures',default=False,action='store_true')
Parse.add_argument('--shape_weights',help='mutation type probabilities for shape changer',\
                   default=(.15, .15, .15, .15, .20, .20), nargs=6)
Myarg = Parse.parse_args()

# genes is number of atoms or list of atoms 
if len(Myarg.genes) == 1:
    genes = tuple(range(Myarg.genes.pop()))
Myarg.genes = genes

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

#no island model
if Myarg.isles == 1:
    Myarg.migration = (0,'Round',None)
try:
    os.path.isfile(Myarg.INP)
except False:
    raise FileNotFoundError("ERROR: Template zmatrix not found")

###------- SETUP

# caller initialization
dry_run = not Myarg.dry_run

if Myarg.xTB:
    if Myarg.Flex:
        flex = "opt"
    else:
        flex = ""
    mycaller = xtb_parser.xTBcaller(template=Myarg.INP, filename=Myarg.NAME, notconv=Myarg.notconv,\
        stopok=Myarg.stopok, run_esc=dry_run,sptype="clust")
else:
    mycaller = gau_parser.gaucaller(template=Myarg.INP, filename=Myarg.NAME, shellname=Myarg.SHELL,\
        notconv=Myarg.notconv, stopok=Myarg.stopok, optstr=Myarg.g16_opt, \
        run_alw=Myarg.always, run_esc=dry_run, sptype="clust")
Xtpl = mycaller.initX

if Myarg.xTB:
    Ctpl = cluster_utils.xtb_cluster(Xtpl, Myarg.INP, Myarg.genes)
else:
    Ctpl = cluster_utils.gau_cluster(Xtpl, Myarg.INP, Myarg.genes)

cmat = cluster_utils.ClashCheck(Ctpl.atoms, Xtpl)

## NB solvent after!!
AT0=0
AT1=1
NAT=3

# cluster mutator initialization
if Myarg.mut_meth == "shapechanger":
    shapechanger = cluster_utils.shapechanger(at0=AT0, at1=AT1, nat=NAT, \
                                  weights=np.array(Myarg.shape_weights,
                                                   dtype='float'),
                                             check=cmat.check_geom)
elif Myarg.mut_meth == "shapechanger2":
    shapechanger = cluster_utils.shapechanger2(at0=AT0, at1=AT1, nat=NAT, \
                                  weights=np.array(Myarg.shape_weights,
                                                   dtype='float'),
                                             check=cmat.check_geom)
else:
    raise ValueError("Use shapechanger or adaptive mutator for molecular clusters")
Myarg.mut_meth = "custom"

# cluster cross_over initialization
if Myarg.co_meth == "custom":
    cross_over = cluster_utils.complex_co(at1=AT1,nat=NAT,eta=5)
else:
    raise ValueError("Use custom cross-over for molecular clusters")

###-------- INITIAL POPULATION    

if Myarg.xTB:
    caller_type = "xTB"
else:
    caller_type = "g16"
mypop = gen_init_pop.init_pop_cluster(Myarg.nchrm, Myarg.INP, Myarg.genes, Myarg.cutoff, \
	Myarg.verbose, Myarg.notconv, Ctpl, cmat, AT0, AT1, NAT, shapechanger, cross_over, mycaller, 
	caller_type, Myarg.xTB)

###------------ EVOLUTION

print("+++ RUNNING GA with settings:")
print(Myarg) 
if Myarg.Flex:
    print("WARNING: doing a geometry optmization for each fitness evaluation")
    
myGA = ga_evolution.GenAlg(last_rank=Myarg.rank, sel_press=Myarg.sel_press, pCo=Myarg.pCo,\
            ppmut=Myarg.ppmut, pcmut=Myarg.pcmut, verbose=Myarg.verbose, co_meth=Myarg.co_meth,\
            fhigh=Myarg.fhigh, tsize=Myarg.tsize, seed=Myarg.upd,\
            mut=Myarg.mut_meth, mfunc=shapechanger, cofunc=cross_over, debug=Myarg.debug)

fitness, bestc, bestID = myGA.Evolve(genotype=None, pop=mypop, niter=Myarg.niter,\
            ffunc=mycaller.fitcalc ,initfit=True, fitkwds=KWDS, LS=Myarg.LS, nisle=Myarg.isles,\
            migr_freq=int(Myarg.migration[0]), mpolicy=Myarg.migration[1], distance=Myarg.migration[2])
     
###---------- CLOSE

outf = open("fitness_" + Myarg.NAME + ".dat","w")
for i in range(Myarg.niter):
    outf.write("{0:5d} {1:15.9f} {2:5d}\n".format(i,fitness[i],bestID[i]))

quit()
