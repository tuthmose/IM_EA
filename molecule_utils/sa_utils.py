import numpy   as np    
import os
import re
import scipy   as sp
import subprocess as sub
import zmatrix as ZM
import ga_utils

# Giordano Mancini Feb 2019

deg2rad = np.pi/180.
rad2deg = 1./deg2rad

class molecule(object):
    """
    Container object that includes a zmatrix instance, an ID
    and attributes to read and write coordinates
    """
    def __init__(self,template,rot_bonds,dihedrals=None):
        self.zmat = ZM.zmat(template,rot_bonds=rot_bonds,fmt="gau")
        self.rot_bonds = rot_bonds
        self.energy = np.nan
        self.ID     = np.nan
        self.ndih = len(self.rot_bonds)
        if dihedrals is not None:
            self.zmat.update_dihedrals(dihedrals, list(range(self.ndih)), update=True)
        self.Coordinates = self.zmat.dump_coords(retc=True)
              
    def set_fitness(self,energy):
        self.energy = energy
        
    def get_fitness(self):
        return self.energy

    def get_ID(self):
        return self.ID

    def set_ID(self,ID):
        if(ID > 0):
            self.ID = ID
        else:
            raise ValueError
    
    def update_dihedrals(self, newdihedrals):
        """
        given a new set of dihedral values update zmatrix
        """
        self.zmat.update_dihedrals(newdihedrals,list(range(self.ndih)),update=True)
                
    def get_chromosome(self):
        l = list()
        for rb in self.rot_bonds:
            rble_bond = (rb[0]-1,rb[1]-1)
            for atom in range(self.zmat.natoms):
                if atom in self.zmat.atom3:
                    r1 = (self.zmat.atom1[atom],self.zmat.atom2[atom])
                    r2 = (self.zmat.atom2[atom],self.zmat.atom1[atom])   
                    if rble_bond in (r1,r2):
                        l.append(self.zmat.dih[atom])
                        break
        dihedrals = np.asarray(l)*rad2deg
        return dihedrals
    
def new_dihedrals_glob(molecule,domain):
    """
    generate new dihedrals with non memory
    """
    nangles = molecule.ndih
    dih = domain(nangles)
    molecule.update_dihedrals(dih)
    X = molecule.Coordinates
    return molecule, X    
    
    
def new_dihedrals_near(molecule,domain,cutoff,flat=False):
    """
    generate a new dihedral configuration
    with coordinates near the current ones
    """   
    dih = molecule.get_chromosome()
    for i in range(molecule.ndih):
        if flat == False:
            #dihedral from normal distrib
            dih[i] = np.random.normal(loc=dih[i], scale=cutoff)
        else:
            # dihedral from flat distrib
            dih[i] = dih[i] + 2.*cutoff*(np.random.rand()-1.)
    molecule.update_dihedrals(dih)
    X = molecule.Coordinates
    return molecule, X
    
def gen_mol(molecule, domain=None, cutoff=None, glob_prob=0.01):
    """
    generate a new molecule by changing dih. values
    or picking entirely new ones (glob_prob)
    """
    if domain == None:
        raise ValueError("Provide domain to gen_mol")    
    if cutoff == None:
        raise ValueError("Provide cutoff to gen_mol")
    
    clash = True
    coin = np.random.rand()
    if coin <= glob_prob:
        while clash:
            molecule, X = new_dihedrals_glob(molecule,domain)
            clash = ga_utils.check_clash(X)
    else:
        while clash:
            molecule, X = new_dihedrals_near(molecule,domain,cutoff)
            clash = ga_utils.check_clash(X)
    return molecule
        
