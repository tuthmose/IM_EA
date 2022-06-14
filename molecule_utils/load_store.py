# Miscellaneous functions to:
# extract energies and coordinates from gaussian log files
# create xtc and pandas dataframe
# print gaussian input files from trajectories
# G Mancini december 19

import bz2
import numpy  as np    
import mdtraj as md
import os
import pandas as pd
import re
import scipy as sp
import scipy.constants
from scipy.spatial.distance import pdist
#
import zmatrix
import gau_parser
import ga_utils
import xtb_parser

bohr2ang = scipy.constants.physical_constants['Bohr radius'][0]*10e9
ang2bohr = 1./bohr2ang
deg2rad = np.pi/180.
rad2deg = 1./deg2rad
circle  = 2.*np.pi

def parse_log(REGX, pat, num, caller, style, molecule, f, log, DF, RB, traj, ferr, count, loadErr):
    # check if termination is normal
    isError = REGX['Error'].search(log)
    if loadErr == False and isError is not None:
        ferr = ferr + 1
        return ferr, count
    elif loadErr is True  and isError is not None:
        ferr = ferr + 1
    elif isError == None:
        pass
    else:
        raise ValueError("Unknown combination of error and loadErr value")
    # check of we did a geometry optimization
    if caller == "g16":
        IsOpt = REGX['opt'].search(log)
    elif caller == "xTB":
        IsOpt = REGX['CnvOk'].search(log)     
    # load geometry
    if caller == ("g16" and IsOpt is not None) and ((loadErr is False) or (loadErr is True and isErr is not None)):
        X = REGX['fgeom'].search(log).group(2)
    elif caller == "g16":
        X = REGX['fgeom_sp'].search(log).group(1)
    elif caller == "xTB" and IsOpt is not None:
        X = REGX['fgeom'].search(log).group(1)
    elif caller == "xTB":
        mod = ''
        if IsOpt:
            mod = '.xtbopt'
        try:
            infile = bz2.open("../Coord/" + pat + "_" + num + mod + ".coord.bz2", "r")
            records = infile.read().decode()
        except:
            infile = open("../Coord/" + pat + "_" + num + mod + ".coord", "r")
            records = infile.read()
        X = REGX['fgeom_sp'].search(records).group(1)
        infile.close()
    else:
        raise ValueError('Unknown combination of caller and opt')
    # save coordinates in numpy array
    data = list(map(lambda x: x.split(),X.splitlines()))
    cartesian = list()
    if caller == "xTB":
        for d in data:
            if IsOpt:
                line = [str(float(i)*bohr2ang) for i in d[:-1]] + [d[-1]]
            else:
                line = [str(i) for i in d[:-1]] + [d[-1]]
            cartesian.append(line)       
    elif molecule is not None and caller == "g16":
        for d in data:
            # cartesian.append(d[3:])
            cartesian.append(list(map(float,d[3:])))
    else:
        for d in data:
            line =  d[3:]
            line.insert(0,d[1])
            cartesian.append(line)
    cartesian = np.asarray(cartesian)
    # get Energy
    if caller == "g16" and style=="DFT":
        Energy = float(REGX['lastE'].findall(log).pop()[0])
    elif caller == "g16":
        Energy = REGX['lastE'].search(log).group(1)
        Energy = float(Energy.split()[1])
    elif caller == "xTB" and IsOpt is not None:
        Energy = REGX['lastE'].search(log).group(3)
    elif caller == "xTB":
        Energy = REGX['singlepE'].search(log).group(1)
    else:
        raise ValueError('Unknown combination of caller and opt')
    Energy = float(Energy)
    # get dipole
    if 'dpt' in DF.columns:
        dipole =  [float(x) for x in REGX['dpall'].search(log).group(1,2,3,4)]

    if molecule is not None:
        if caller == "xTB":
            cartesian = np.array(cartesian[:, :3], dtype=float) 
        molecule.zmat.update_internal_coords(cartesian)
        dihedrals = molecule.get_chromosome()
        hdr = "{0:6d}\n{1:12.7f}\n".format(molecule.zmat.natoms,Energy)
        frame = hdr + \
          molecule.zmat.dump_coords(newdih=None,select=None,retc=True,zstr=True)
        L = DF.shape[0]
        DF.loc[L,RB] = dihedrals
        DF.loc[L,'energy'] = Energy
        DF.loc[L,'filename'] = f
        if 'dipole moment' in DF.columns:
            Dip = float(Dipole.findall(log).pop().split()[-1])
            DF.loc[L,'dipole moment'] = Dip
    else:
        L = DF.shape[0]
        DF.loc[L,'energy'] = Energy
        DF.loc[L,'filename'] = f                    
        if 'dpt' in DF.columns:
            DF.loc[L,'dpt'] = dipole[3]
            DF.loc[L,'dpx'] = dipole[0]
            DF.loc[L,'dpy'] = dipole[1]
            DF.loc[L,'dpz'] = dipole[2]
        natoms = len(cartesian)
        hdr = "{0:6d}\n{1:12.7f}\n".format(natoms,Energy)
        frame = hdr
        if caller == "g16":
            for line in cartesian:
                frame = frame + '{:4s} {:10.7s} {:10.7s} {:10.7s}\n'.\
                    format(line[0],line[1],line[2],line[3])
        else:
            for line in cartesian:
                frame = frame + '{:4s} {:10.7s} {:10.7s} {:10.7s}\n'.\
                    format(line[-1],line[0],line[1],line[2])            
    count += 1
    traj.write(frame)
    return ferr, count
    
def get_from_log(pat, molecule=None, RB=None, mu=False, style=None, caller=None, loadErr=False):
    """
    given a regex, parse log files and if Normal termination is found:
    - put all coordinates in a xyz trajectory
    - load energy dipole moment and rotatable bonds in pandas dataframe
    """
    files = sorted(os.listdir())
    rgx = re.compile(pat)
    traj = open("traj_"+pat+".xyz","w")
    #TODO: auto number of digits
    npat = re.compile(r'_([0-9][0-9][0-9][0-9][0-9])\.')
    #npat = re.compile(r'_([0-9][0-9][0-9][0-9])\.')
    #npat = re.compile(r'_([0-9][0-9][0-9])\.')
    
    #type of output
    if molecule is None:
        # cluster object only save cartesian data
        cols = ['energy', 'filename']
    else:
        #create data frame
        cols = list()
        cols = RB + ['energy', 'filename']
    if mu:
        cols += ['dpt', 'dpx', 'dpy', 'dpz']

    DF = pd.DataFrame(data=None, columns=cols) 
        
    #regular expressions
    if caller == "g16":
        REGX = gau_parser.output_style(style)
    elif caller == "xTB":
        REGX = xtb_parser.output_style()
    else:
        raise ValueError("get_from_log: wrong or not specified caller")
        
    count = 0
    ferr  = 0
    for f in files:
        try:
            if rgx.search(f) is not None:
                num = npat.search(f).group(1)
                if f[-3:]=="bz2":
                    log_file = bz2.open(f,"r")
                    log  = log_file.read().decode()
                else:
                    log_file = open(f,"r")
                    log  = log_file.read()
                ferr, count = \
                parse_log(REGX, pat, num, caller, style, molecule, f, log, DF, RB, traj, ferr, count, loadErr)
                log_file.close()
        except:
            pass

    traj.close()
    return count, ferr, DF 
    
def write_bunch_inp(**kwargs):
    """
    given a gaussian template, a trajectory and frame
    write corresponding gaussian input files
    rot_bonds are passed but not really useful
    since 2nd level opts are fully flexible
    """
    tpl       = kwargs['tpl']
    prefix    = kwargs['prefix']
    traj      = kwargs['traj']
    rot_bonds = kwargs['rot_bonds']
    frames    = kwargs['frames']
    shell     = kwargs['shell']
    sptype    = kwargs['sptype']
    zm_in     = kwargs['zm_in']
    pdbs      = kwargs['pdbs']
    pdb_tpl   = kwargs['pdb_tpl']
    #
    zmat = zmatrix.zmat(tpl,rot_bonds,fmt="gau")
    #instance gau_parser
    mycaller = gau_parser.gaucaller(template=tpl,filename=prefix,shellname=shell,\
        sptype=sptype)
    nfile = 0
    if frames == None:
        frames = range(traj.n_frames)
    for f in range(len(frames)):
        n = '{0:>03}'.format(f)
        name = prefix + n
        coords = 10.*traj[f].xyz[0]
        zmat.update_internal_coords(coords)
        if sptype == "mol":
            ZM = zmat.dump_coords(retc=True,zstr=True)
        elif zm_in is True:
            ZM = zmat.write_zmat(False)
        elif sptype == "clust":
            ZM = zmat.dump_coords()        
        if f in pdbs:
            zmat.write_file(name+".pdb","pdb",template=pdb_tpl)
        mycaller.writeinp(ZM, name+".com", False)
        nfile += 1
    return nfile
    
def RMSD_filter(cutoff, mw, RMSD):
    """
    calc (mass weighted) RMSD matrix and filter
    unique structures using cutoff
    """
    ### check cartesian data
    RMSD = np.empty(0)
    for i in range(traj.n_frames):
        rmsd_frame = md.rmsd(traj,traj,frame=i)
        RMSD = np.append(RMSD,rmsd_frame)
    RMSD.shape = ((nfiles,nfiles))
    print("Saving RMSD matrix in ",)
    np.savetxt(Myarg.out+"_rmsd.dat",RMSD)

    print("Checking unique structures in RMSD matrix")
    R = RMSD.shape[0]
    C = RMSD.shape[1]
    uniq = np.ones(R,dtype='int')

    #convert from ang to nm; mdtraj uses nm
    Myarg.cutoff = Myarg.cutoff/10.

    for row in range(R):
        for col in range(row+1,C):
            overlap = np.asarray(RMSD[row,col:] < Myarg.cutoff)#.nonzero()[0]
            current = uniq[col:]
            current[overlap] = 0
            
def dih_cutoff(cutoff, DF):
    """
    return a new DF with unique dihedrals beyond cutoff
    selecting lowest energies
    """
    L = len(DF.index)
    for r in range(L):
        within = 0
        row = DF.loc[r,Myarg.rot_bonds].values
        delta = np.abs(row - arr)
        overlap = np.asarray(delta < Myarg.dih_cutoff).nonzero()[0]
        within = len(overlap)
        if within == ncols:
            if energy < DF.loc[r,'energy']:
                DF.loc[r,Myarg.rot_bonds] = arr
                DF.loc[r,'energy'] = energy
                DF.loc[r,'count']  += 1           
                DF.loc[r,'filename'] = f
            return uniq
    DF.loc[L,Myarg.rot_bonds] = arr
    DF.loc[L,'energy'] = energy
    DF.loc[L,'count'] = 1
    DF.loc[L,'filename'] = f        
    uniq += 1
    return uniq

def compute_distances(xyz, pairs):
    """
    Given array nframes x natoms x 3
    and tuple of tuples of pairs
    computes distance between pairs 
    for all frames
    """
    nframes = xyz.shape[0]
    D = np.empty((nframes, len(pairs)))
    for frame in range(nframes):
        for p, P in enumerate(pairs):
            D[frame, p] = np.linalg.norm(xyz[frame, P[0]] - xyz[frame, P[1]])
    return D

def compute_angles(xyz, pairs, dmax=None):
    """
    Given array nframes x natoms x 3
    and tuple of tuples of pairs
    computes angles between pairs 
    of tuples for all frames
    """
    nframes = xyz.shape[0]
    npp = len(pairs)
    pairs_of_pairs = list()
    for pi in range(npp):
        for pj in range(pi+1, npp):
            pairs_of_pairs.append((pairs[pi], pairs[pj]))
    A = np.empty((nframes, npp * (npp-1)//2 ))
    for frame in range(nframes):
        for i, ppi in enumerate(pairs_of_pairs):
            v1 = xyz[frame, ppi[0][0]] - xyz[frame, ppi[0][1]]
            v2 = xyz[frame, ppi[1][0]] - xyz[frame, ppi[1][1]]
            nv1 = np.linalg.norm(v1)
            nv2 = np.linalg.norm(v2)
            alpha = np.dot(v1, v2)/(nv1 * nv2)
            A[frame, i] = alpha
    return A, npp

def reorder_traj(**kwargs):
    """
    given a PDB trajectory and a key
    write a new trajectory with order
    given in key
    """
    traj = kwargs['traj']
    top = kwargs['top']
    key  = kwargs['key']
    # open key file
    kf = open(key, 'r')
    lines = kf.readlines()
    atom_order = list()
    for line in lines:
        record = line.split()
        if record[0] == '#':
            continue
        atom_order.append(int(record[1]))
    atom_order = np.array(atom_order) - 1
    # reorder
    newxyz = list()
    for f in range(traj.n_frames):
        myarray = traj.xyz[f][atom_order]
        newxyz.append(myarray)
    newxyz = np.asarray(newxyz)
    newtraj = md.Trajectory(newxyz, top)
    return newtraj
    
