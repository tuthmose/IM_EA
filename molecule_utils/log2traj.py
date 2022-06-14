import argparse as argp
import numpy as np   
import mdtraj as md
import os
import pandas as pd
import re
import ga_utils
import load_store

Parse = argp.ArgumentParser(description='Search for G16 log files, convert to traj and load uniq structures')
# template and gaussian settings
Parse.add_argument('-c','--cutoff',help='RMSD cutoff to consider equal two structures (angstrom); if 0 do no calculate RMSd',\
    default=0.5,type=float)
Parse.add_argument('-d','--dih_cutoff',help='dihedral cutoff to consider equal two \
    torsions',default=0.,type=float)
Parse.add_argument('-o','--out',help='output name frame prefix',default=None,type=str)
Parse.add_argument('-S','--sp',help='is a single point run',default=True,action='store_false')
Parse.add_argument('-r','--regex',help='regular expression for file names',default=None,type=str)
Parse.add_argument('-I','--INP',help='template gaussian input file',default='template.com',type=str)
Parse.add_argument('-R','--rot_bonds',help='rotatable bonds',default=None,type=int,nargs='+')
Parse.add_argument('-s','--style',help='output style',default="DFT",type=str)
Parse.add_argument('-mw',help='mass weighted RMSD',default=False,action='store_true')
Parse.add_argument('-T','--top',help='topology in PDB',default=None,type=str)
Myarg = Parse.parse_args()
print(Myarg)
deg2rad = np.pi/180.
rad2deg = 1./deg2rad
pd.set_option('precision', 8)

if Myarg.out is None:
    print("WARNING: missing output name prefix; RMSD matrix will not be saved")

# check input is existing
if Myarg.regex is None:
    raise ValueError
try:
    os.path.isfile(Myarg.INP)
except False:
    raise FileNotFoundError("ERROR: Template zmatrix not found")

# create a molecule
mytuple = list()
for i in range(0,len(Myarg.rot_bonds),2):
    mytuple.append((Myarg.rot_bonds[i],Myarg.rot_bonds[i+1]))
Myarg.rot_bonds = mytuple
ncols = len(Myarg.rot_bonds)
molecule = ga_utils.linear_molecule(Myarg.INP,Myarg.rot_bonds)
                                     
Nok, Nerror, DF = load_store.get_from_log(Myarg.regex,molecule,Myarg.rot_bonds,Myarg.style,Myarg.out,opt=Myarg.sp)
print("Success optimizations: {0:d}\n error optimizations {1:d}".format(Nok, Nerror))
print("Loading xyz traj ...")
traj = md.load("traj_" + Myarg.out + ".xyz",top=Myarg.top)
print(traj)
traj2 = traj.superpose(traj, frame=0, atom_indices=None, ref_atom_indices=None, parallel=True)
traj2.save_pdb("traj_" + Myarg.out + ".pdb")
del traj
traj = traj2
print("Saving dataframe")
DF.to_pickle("df_"+Myarg.out+".data")

if Myarg.cutoff != 0.:
    RMSD, uniq = load_store.RMSD_filter(Myarg.cutoff,Myarg.mw,traj)

    print("Found {0: 6d} unique structures in RMSD matrix : ".format(len(uniq.nonzero()[0])))
    for i, j in enumerate(uniq):
        if j != 0:
            print(i)
            
if Myarg.dih_cutoff != 0.:
    raise ValueError

quit()
