import bz2
import numpy   as np    
import os
import re
import scipy   as sp
import subprocess as sub
import zmatrix as ZM
from scipy.spatial.distance import pdist

# Giordano Mancini Dec 2018

deg2rad = np.pi/180.
rad2deg = 1./deg2rad

class specimen:
    """
    Container object that includes a zmatrix instance, an ID
    and attributes to allow genetic operations
    - genes (int tuple) contains atoms (zmatrix lines) that will undergo
    mutation and crossover
    - alleles (float tuple) contains the initial dihedral values)
    - ID to identify QM corresponding QM calculation
    """
    def __init__(self):
        pass
        
    def set_chromosome(self):
        pass

    def get_coordinates(self):
        pass

    def set_energy(self,fitness):
        self.fitness = fitness
        
    def get_energy(self):
        return self.fitness

    def get_ID(self):
        return self.ID

    def set_ID(self,ID):
        if(ID > 0):
            self.ID = ID
        else:
            raise ValueError

    #aliases
    def get_chromosome(self):
        return self.get_coordinates()

    def get_fitness(self):
        return self.get_energy()

    def set_fitness(self, fitness):
        self.set_energy(fitness)

class linear_molecule(specimen):
    """
    to search in torsional space
    """
    def __init__(self,template,genes,alleles=None,cutoff=0.8):
        self.zmat = ZM.zmat(template,rot_bonds=genes,fmt="gau")
        self.genes = genes
        self.cutoff = cutoff
        if np.any(alleles):
            self.set_chromosome(alleles)
        #self.Coordinates = self.zmat.dump_coords(retc=True)
        self.fitness = np.nan
        self.ID      = np.nan
        self.ngene = len(self.genes)
        clash = self.zmat.check_clash(cutoff)
        if clash:
            print("Clash ",self.zmat.mindist(True),self.get_chromosome())
            self.zmat.write_file("test.xyz",fmt="xyz")
            raise ValueError("Clash in coordinates")
        
    def set_chromosome(self,newdihedrals,genes=None):
        """
        given a new set of dihedral values
        (alleles) for genes, update zmatrix
        """
        if genes is None:
            genes = list(range(len(self.genes)))
        elif len(genes) != len(newdihedrals):
            raise ValueError("Specimen.set_chromosomes: different number of genes and allele provided")  
        self.zmat.update_dihedrals(newdihedrals,genes,update=True)
        self.alleles = self.get_chromosome()
        
    def get_coordinates(self):
        l = list()
        for rb in self.genes:
            rble_bond = (rb[0]-1,rb[1]-1)
            for atom in range(self.zmat.natoms):
                if atom in self.zmat.atom3:
                    r1 = (self.zmat.atom1[atom],self.zmat.atom2[atom])
                    r2 = (self.zmat.atom2[atom],self.zmat.atom1[atom])   
                    if rble_bond in (r1,r2):
                        l.append(self.zmat.dih[atom])
                        break
        alleles = np.asarray(l)*rad2deg
        return alleles

    def dump_coords(self, **kwargs):
        return self.zmat.dump_coords(**kwargs)

    def update_internal_coords(self, xyz):
        self.zmat.update_internal_coords(xyz)
    

class cyclic_molecule(specimen):
    pass
                    
class dihedral_space:
    """
    define dihedral space at given resolution and
    return a number of values sampled from normal
    distributions
    """
    def __init__(self,resolution,scale=5.0):
        self.angles =  np.arange(-180.0,181.0,resolution)
        self.angles =  np.arange(-180.0,181.0,resolution)
        self.scale = np.ones(len(self.angles))*scale
        
    def gen_normal(self):
        return np.random.choice(np.random.normal(loc=self.angles,scale=self.scale))
        
    def __call__(self,n):
        out = np.zeros(n)
        for i in range(n):
            out[i] = self.gen_normal()
            if out[i]<-180.:
                out[i] = -180.
            elif out[i] > 180.:
                out[i] = 180.
        return out
    
def get_all_ID(zmatlist):
    I = list()
    for z in zmatlist:
        I.append(z.get_ID())
    return I

def get_all_fitness(zmatlist):
    """
    given a list of zmat instances, get all fitness values
    and return array
    """
    F = list()
    for z in zmatlist:
        F.append(z.get_fitness())
    return F

def get_all_chromosomes(zmatlist):
    """
    given a list of zmat instances,
    get all alleles and return array
    """
    C = list()
    for z in zmatlist:
        C.append(z.get_chromosome())
    return C

def get_all_coordinates(zmatlist):
    """
    given a list of zmat instances,
    get all coordinates and return 
    (n,natoms,3) array
    """
    C = list()
    for z in zmatlist:
        C.append(z.zmat.dump_coords(retc=True))
    return C

def dump_all_coords(zmatlist,name):
    """
    given a list of zmat instances,
    write an xyz trajectory
    """
    myname = "tmp_ga_zmat_"
    for i, z in enumerate(zmatlist):
        z.zmat.write_file(myname+str(i)+".xyz",fmt="xyz")
    args = 'cat tmp_ga_zmat_*.xyz > traj_ga_zmat.xyz; rm -f tmp_ga_zmat_*.xyz'
    sub.run(args, stderr=sub.PIPE,shell=True)
    os.rename('traj_ga_zmat.xyz',name)

def convert_to_traj(regex,template,out):
    """
    search for zmatrix file with regex,
    load and create cartesian .pdb trajectory
    with template
    """
    rgx = re.compile(regex)
    files = os.listdir()
    traj = open(out,"w")
    for f in files:
        if rgx.search(f) is not None:
            myzm  = ZM.zmat(f,fmt="zmat")
            myxyz = myzm.dump_coords(retc=True,zstr=True)
            traj.write(myxyz)
    traj.close()
    
def find_energy(regex,energy):
    """
    given regex and energy value find_
    corresponding files
    """
    conv = 1e-7
    rgx = re.compile(regex)
    files = os.listdir()
    okfiles = list()
    energy = float(energy)
    n = 0
    lastE = re.compile(r'SCF Done:\s+E\(\w+\)\s=\s+((\s|-)\d+\.\d+(E-\d+|\s))\s')
    for f in files:
        if rgx.search(f) is not None:
            ff = open(f,"r")
            E = lastE.findall(ff.read())
            ff.close()
            if len(E) > 0:
                E = float(E.pop()[0])
                if abs(E - energy) <= conv:
                    okfiles.append(f)
                    n = n+1
    return okfiles, n
