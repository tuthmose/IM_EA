from math import ceil
from mendeleev import element
import numpy as np
import re
import scipy as sp

from numpy.linalg import norm

# custom modules
from ga_utils import *
from gau_parser import input_style as gau_regexp
from xtb_parser import input_style as xtb_regexp
import quaternion_utils

# Giordano Mancini Jan 2019
# minor Feb 2021

deg2rad = np.pi/180.
rad2deg = 1./deg2rad
B2A = 0.529177210903

def check_clash(coords, cutoff=0.7):
    S = sp.spatial.distance.pdist(coords)
    if S.min() < cutoff:
        return True
    else:
        return False

class ClashCheck():
    """
    Simple class to check the connectivity of a new geometry wrt of a reference
    structure
    """
    def __init__(self, atnums, refcrd, thrs=0.4, tol=0.):
        """
        atnums: list of atomic number
        refcrd: reference atomic crd in Angstrom as natom x 3 matrix
        thrs: threshold to compute the connectivity matrix
        tol: tolerance right now a place holder argument
        """
        self._atn = atnums
        self._refcrd = refcrd
        self._thrs = thrs
        self._tol = tol
        # use the covalent radii to evaluate the connectivity
        self._ard = np.array([element(x).covalent_radius_pyykko/100 for x in self._atn])
        self._compute_ardmat()
        self._updatecref()

    def _compute_ardmat(self):
        """
        compute the sum of covalence radius matrix
        """
        tmp = self._ard[:, np.newaxis] + self._ard[np.newaxis,:]
        self._ardmat = tmp

    def _compute_conn(self, crd):
        """
        evaluate the connectivity of a given coordinate based on the atomic
        number of the reference structure
        crd: coordinates in Angstrom as natom x 3 matrix
        Return:
            A boolen matrix 
        """
        dmat = sp.spatial.distance_matrix(crd, crd)
        ardsum = self._ardmat + (self._thrs + self._tol)
        ardsum[np.diag_indices(self._ard.shape[0])] = 0.
        return (ardsum - dmat) > 0

    def _updatecref(self):
        """
        update the connectivity of the reference structure
        """
        self._cref = self._compute_conn(self._refcrd)

    def set_thrs(self, thrs):
        """
        set a different threshold to evaluate connectivity
        """
        self._thrs = thrs
        self._updatecref()

    def set_tol(self, tol):
        """
        set a different tolerance (NB: not useful right now, use set_thrs
        instead)
        """
        self._tol = tol
        self._updatecref()

    def check_geom(self, crd):
        """
        Compare the given coordinate with respect of the reference geometry.
        returns True is there are changes in geometry, and False if the
        geometry are the same (To be interchangeable with check_clash)
        crd: the coordinate to check in angstrom (natoms x 3)
        return:
            bool
        """
        new_conn = self._compute_conn(crd)
        return (self._cref ^ new_conn).any()

    def get_conmat(self, crd=None):
        """
        Returns the connectivity matrix of a given coordinates.
        If None is given the returns the connectivity matrix of the reference
        """
        if crd is None:
            return self._cref
        else:
            return self._compute_conn(crd)

class cluster(specimen):
    """
    Container object that includes cartesian coordinates, an ID
    and attributes to allow genetic operations.
    Used to run GA on molecular clusters instead of single 
    molecules. 
    NB ALL VALUES IN ANGSTROM
    - genes (int tuple) contains atoms (zmatrix lines) that will undergo
    mutation and crossover
    - coordinates: np.ndarray ngenes x 3 (to allow for fixed atoms)
    - ID to identify QM corresponding QM calculation
    The following methods are borrowed from ga_utils.specimen:
    - set_fitness
    - get_fitness
    - get_ID
    - set_ID
    """       
    def set_chromosome(self,coords):
        """
        assign new coordinates to non fixed atoms
        """
        assert coords.shape == (self.ngene, 3)
        self.X = np.copy(coords)
        #self.X[self.genes] = np.copy(coords)
    
    def get_chromosome(self):
        """
        get coordinates
        """
        return self.X

    def get_coordinates(self):
        return self.X   

    def set_coordinates(self, coords):
        self.X = np.copy(coords)    
    
    def __init__(self, coords, template, genes=None):
        assert coords.shape[1] == 3
        if genes is not None:
            assert len(genes) == coords.shape[0]
            self.genes = genes
        else:
            self.genes = tuple(range(coords.shape[0]))
        self.fitness = np.nan
        self.ID      = np.nan
        if check_clash(coords):
            print(coords)
            print(sp.spatial.distance.pdist(coords).min())
            raise ValueError("Clash in coordinates")
        self.ngene = len(self.genes)
        self.X = np.copy(coords)
        # set regexps and parse template
        self.read_template(template)
        
class gau_cluster(cluster):        
    def read_template(self, template):
        REGX = gau_regexp()
        templ = open(template,"r")
        templL = templ.read()  
        records = (REGX['coords'].search(templL).group(2)).split("\n")
        nrec = len(records)
        self.atoms = list()
        self.atmass = list()
        # data in the input line other than Z and coordinates (e.g. fragment, charge)
        self.add_info = list()
        self.optat = list()
        self.oniom_layers = list()
        for r in records:
            if len(r) > 0:
                line = r.split()
                #does not work with Z>99 (fermium and heavier)
                if len(r) == 5:
                    oa = str(r[1])
                else:
                    oa = "0"
                self.optat.append(oa.rjust(6))
                #delete any +/- for monoletter symbols
                at = line[0]
                field0 = REGX['alphac'].search(at)
                atx = field0.group(1)
                info = field0.group(2)
                try:
                #atomic number given
                   atx = int(atx) 
                except:
                   pass
                self.atoms.append(atx)
                self.atmass.append(element(atx).mass)
                if len(info) > 0:
                    self.add_info.append(info)
                # save oniom oniom_layers
                if REGX['oniom'].search(line[-1]) is not None:
                    self.oniom_layers.append(line[-1])
                else:
                    self.oniom_layers.append("  ")
        templ.close()

    def dump_coords(self):
        xyz = ""
        if len(self.add_info) > 0:
            for i in range(self.X.shape[0]):
                field0 = self.atoms[i]+self.add_info[i]+self.optat[i]
                xyz = xyz + '{:40s} {:10.7f} {:10.7f} {:10.7f} {:2s}\n'\
                    .format(field0,self.X[i][0],self.X[i][1],self.X[i][2],self.oniom_layers[i])
        else:
            for i in range(self.X.shape[0]):
                xyz = xyz + '{:4s} {:10.7f} {:10.7f} {:10.7f} {:2s}\n'\
                    .format(str(self.atoms[i]),\
                        self.X[i][0], self.X[i][1], self.X[i][2], self.oniom_layers[i])
        return xyz

class xtb_cluster(cluster):
    def read_template(self, template):
        REGX = xtb_regexp()
        templ = open(template,"r")
        templL = templ.read()
        #print(templL, REGX['coords'], REGX['coords'].search(templL))
        records = (REGX['coords'].search(templL).group(1)).split("\n")
        nrec = len(records)
        self.atoms = list()
        self.atmass = list()
        for r in records:
            if len(r) > 0:
                line = r.split()
                #delete any +/- for monoletter symbols
                atx = line[-1]
                try:
                #atomic number given
                   atx = int(atx) 
                except:
                   pass
                self.atoms.append(atx)
                self.atmass.append(element(atx).mass)
        templ.close()
    
    def dump_coords(self):
        xyz = ""
        for i in range(self.X.shape[0]):
            xyz = xyz + '{:10.7f} {:10.7f} {:10.7f} {:4s}\n'\
                .format(self.X[i][0], self.X[i][1], self.X[i][2], self.atoms[i])
        return xyz
    
def write_xyz(atoms, coords, fname):
    out = open(fname,"w")
    xyz = "{0:6d}\n\n".format(coords.shape[0])
    for i, at in enumerate(atoms):
        if isinstance(at, int):
            atom = str(at)
        else:
            atom = at
        xyz = xyz + '{:4s} {:10.7f} {:10.7f} {:10.7f}\n'\
            .format(atom,coords[i][0],coords[i][1],coords[i][2])
    out.write(xyz)
    out.close()

def rattle(coords, at1, nat, ave, std, maxtry, check=check_clash):
    """
    given a molecular cluster, move all atoms from at 1 randomly
    """
    newc = np.copy(coords)
    clash = True
    ntry  = 0
    nx = coords[at1:,:].shape
    while clash and ntry < maxtry:
        mov = np.random.normal(loc=ave, scale=std, size=nx)
        newc[at1:] = coords[at1:] + mov
        clash = check(newc)
        ntry += 1
    if ntry >= maxtry and clash:
        newc = np.copy(coords)
    return newc

def reflect(vec, vn):
    """
    reflec vector vec by plane normal to vn
    """
    dp = 2.*(np.dot(vec, vn) / np.dot(vn, vn))*vn
    return vec - dp

def mirror_mol(coords, at1, nat, maxtry, ext_cutoff, check=check_clash):
    """
    given a set of coordinates n x 3
    reflect each by the plane normal
    to a random created vector
    """
    newc = np.copy(coords)
    mol = list(range(at1, coords.shape[0], nat))
    at = np.random.choice(mol)
    clash = True
    ntry  = 0
    while clash and ntry < maxtry:
        vn = quaternion_utils.random_vers()
        for at in range(at1, at1 + nat):
            newc[at] = reflect(coords[at], vn)
        clash = check_clash(newc, ext_cutoff)
        clash = clash | check(newc)
        ntry += 1
    if ntry >= maxtry and clash:
        newc = np.copy(coords)
    return newc

def rotate_coords(coords, at1, nat, alpha, sigma, maxtry, atmass,
                  check=check_clash):
    """
    given a set of coordinates n*3
    create a quanternion and 
    rotate the coordinates
    starting from atom at1, considering
    molecules of nat atoms
    """
    newc = np.copy(coords)
    natoms = coords.shape[0]
    mol = list(range(at1, coords.shape[0], nat))
    clash = True
    ntry  = 0
    while clash and ntry < maxtry:
        for at in range(at1, natoms, nat):
            if sigma != 0.:
                angle = np.random.normal(alpha, sigma)
            else:
                angle = alpha[at//nat]
            vec = quaternion_utils.random_vers()
            Q = quaternion_utils.axis_angle(angle, vec)
            X = coords[at:at+nat]
            C = np.average(X, axis=0, weights=atmass[at:at+nat])
            newc[at:at+nat] = C + quaternion_utils.quat_action_mat(Q, X - C)
        clash = check(newc)
        ntry += 1
    if ntry >= maxtry and clash:
        newc = np.copy(coords)
    return newc

def swapmol(coords, at1, nat, maxtry, ext_cutoff, check=check_clash):
    """
    swap two random ligands
    """
    newc = np.copy(coords)
    natoms = coords.shape[0]
    mol = list(range(at1, coords.shape[0], nat))
    clash = True
    ntry  = 0
    atoms = tuple(range(at1, natoms,nat))
    while clash and ntry < maxtry:
        at_1 = np.random.choice(atoms)
        remaining = tuple(set(atoms).difference([at_1]))
        at_2 = np.random.choice(remaining)
        tmp = coords[at_2: at_2 + nat]
        newc[at_2: at_2 + nat] = newc[at_1: at_1 + nat]
        newc[at_1: at_1 + nat] = tmp
        clash = check_clash(newc, ext_cutoff)
        # check topology
        clash = clash | check(newc)
        ntry += 1
    if ntry >= maxtry and clash:
        newc = np.copy(coords)
    return newc

def orbit(coords, at1, nat, alpha, sigma, maxtry,
          atmass, check=check_clash):
    newc = np.copy(coords)
    natoms = coords.shape[0]
    mol = list(range(at1, coords.shape[0], nat))
    clash = True
    ntry  = 0
    atoms = tuple(range(at1, natoms, nat))
    while clash and ntry < maxtry:
        at = np.random.choice(atoms)
        if sigma != 0:
            angle = np.random.normal(alpha, sigma)
        else:
            angle = alpha[at//nat]
        vec = quaternion_utils.random_vers()
        Q = quaternion_utils.axis_angle(angle, vec)
        X = coords[at:at+nat]
        C = np.average(X, axis=0, weights=atmass[at:at+nat])
        C2 = quaternion_utils.quat_action(Q, C)
        newc[at:at+nat] = X - C + C2
        clash = check(newc)
        ntry += 1
    if ntry >= maxtry  and clash:
        newc = np.copy(coords)
    return newc

def orbit2(coords, at1, nat, alpha, sigma, maxtry,
          atmass, check=check_clash):
    newc = np.copy(coords)
    natoms = coords.shape[0]
    # mol = list(range(at1, coords.shape[0], nat))
    # Translate in the fix centre of mass
    czero = np.average(newc, axis=0, weights=atmass[:at1])
    newc -= czero
    clash = True
    ntry  = 0
    atoms = tuple(range(at1, natoms, nat))
    while clash and ntry < maxtry:
        at = np.random.choice(atoms)
        angle = np.random.normal(alpha, sigma)
        vec = quaternion_utils.random_vers()
        Q = quaternion_utils.axis_angle(angle, vec)
        X = coords[at:at+nat]
        C = np.average(X, axis=0, weights=atmass[at:at+nat])
        for mul in range(6):
            # try to move away
            C *= (1 + 0.1*mul)
            C2 = quaternion_utils.quat_action(Q, C)
            newc[at:at+nat] = X - C + C2
            if not check(newc):
                break
        clash = check(newc)
        ntry += (1 + mul)
    if ntry >= maxtry  and clash:
        newc = np.copy(coords)
    else:
        newc += czero
    return newc

def displace(coords, at1, nat, ave, std, maxtry, check=check_clash):
    newc = np.copy(coords)
    natoms = coords.shape[0]
    mol = list(range(at1, coords.shape[0], nat))
    clash = True
    ntry  = 0
    atoms = tuple(range(at1, natoms,nat))
    while clash and ntry < maxtry:
        at = np.random.choice(atoms)
        if std != 0:
            const = 1.5*np.random.normal(loc=ave, scale=std)
        else:
            const = ave[at//nat]
        mov = const * np.ones((nat, 3))
        newc[at:at+nat] = coords[at:at+nat] + mov
        clash = check(newc)
        ntry += 1
    if ntry >= maxtry and clash:
        newc = np.copy(coords)
    return newc        

class shapechanger:
    def __init__(self,**kwargs):
        """
        apply to a population of molecular clusters 
        a mutation chosen randomly among the following
        types:
            - rattle: apply a gaussian displacement 
            to selected atoms
            - mirror: select a molecule in the complex
            and try to reflect it along a mirror wo creating
            collisions
            - rotate: each molecule in the cluster is rotated 
            by N(alpha,sigma) along a random axis
        """
        prop_defaults = {
            "alpha"   : 10.,
            "sigma"   : 5.,
            "at0"     : 0,            
            "at1"     : 1,
            "nat"     : 3,
            "ave"     : 0.15, 
            "std"     : 0.075,
            "maxtry"  : 10,
            "ext_cutoff"  : 1.5,
            "weights" : (.20, .10, .20, .10, .20, .20),
            "check": check_clash,
            "rmsd_insertion": 0.2
            }
        for (prop, default) in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))
        self.alpha = self.alpha*deg2rad
        self.sigma = self.sigma*deg2rad
        modes = ("rattle", "mirror", "rotate", "swap", "orbit", "displace")
        self.mtype = dict(zip(modes,self.weights))
    
    def change_shape(self, chrm, atmass):
        """
        apply one of the mutation operators
        """
        fsum  = 0.
        method_coin = np.random.rand()
        for mode in self.mtype.keys():
            fsum = fsum + self.mtype[mode]
            if fsum >= method_coin:
                break
        if mode == "mirror":
            tmpX = mirror_mol(chrm, self.at1, self.nat,
                              self.maxtry, self.ext_cutoff,
                              self.check)
        elif mode == "rattle":
            tmpX = rattle(chrm, self.at1, self.nat, self.ave,
                          self.std, self.maxtry, self.check)
        elif mode == "rotate":
            tmpX = rotate_coords(chrm, self.at1, self.nat, self.alpha,
                                 self.sigma, self.maxtry, atmass,
                                 self.check)
        elif mode == "swap":
            tmpX = swapmol(chrm, self.at1, self.nat, self.maxtry,
                           self.ext_cutoff, self.check)
        elif mode == "orbit":
                tmpX = orbit(chrm, self.at1, self.nat, self.alpha, self.sigma,
                             self.maxtry, atmass, self.check)
        elif mode == "displace":
            tmpX = displace(chrm, self.at1, self.nat, self.ave, self.std,
                            self.maxtry, self.check)
        else:
            raise ValueError("Unknown mutation type")
        return tmpX
    
    def __call__(self, pop, prob, newp):
        mutated = list()
        atmass = pop.specimens[0].atmass
        for i,chrm in enumerate(pop.chromosomes):
            coin = np.random.rand()
            if coin <= prob:
                tmpX = self.change_shape(chrm, atmass)
                if self.check(tmpX) is False:
                    pop.chromosomes[i] = tmpX
                    pop.specimens[i].set_chromosome(tmpX)
                    mutated.append(i)
        return mutated
    
class shapechanger2(shapechanger):
    """
    try to mutate one or two genes
    """
    def __call__(self, population, prob, newp):
        mutated = list()
        atmass = population.specimens[0].atmass
        for i,chrm in enumerate(population.chromosomes):
            coin = np.random.rand()
            if coin <= prob:
                mtype = np.random.rand()
                #try one or two mutations
                if mtype < 0.5:
                    tmpX = self.change_shape(chrm, atmass) 
                else:
                    tmpY = self.change_shape(chrm, atmass) 
                    tmpX = self.change_shape(tmpY, atmass) 
                if self.check(tmpX) is False:
                    #check diversity
                    Q = quaternion_utils.FitQ(chrm,tmpX,weights=np.asarray(atmass))
                    tmpY = quaternion_utils.\
                        rotateMQ(tmpX,quaternion_utils.quat_cng(Q))
                    rmsd = quaternion_utils.rmsd(chrm,tmpY)
                    if rmsd > self.rmsd_insertion:
                        population.chromosomes[i] = tmpX
                        population.specimens[i].set_chromosome(tmpX)
                        mutated.append(i)
        return mutated


class complex_co:
    def __init__(self,**kwargs):
        """
        Specialized crossover operator for metal clusters.
        Given two clusters (parent1 and parent2) for
        each molecule of nat atoms starting from at1
        search the nearest molecule in parent2 and interpolate
        coordinates.
        The coordination number is determined by parent 1.
        Interpolation is done in a SBX like fashion.
        For each interpolated molecule check that it does
        not clash with other molecules (cutoff1) and then try the next one.
        At the end, check all vs all atoms (cutoff2).
        """
        prop_defaults = {
            "SBX"   : True,
            "eta"   : 5,
            "at0"   : 0,
            "at1"   : 1,
            "nat"   : 3,
            "alpha" : 0.75,
            "ext_cutoff"  : 3.1, #1.5,
            "int_cutoff"  : 0.7,
            "rmsd_cutoff" : 1.0,
            "check": check_clash
            }
        for (prop, default) in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))        
    
    def __call__(self, dummyalpha, p0, p1):
        """
        Call crossover and try to interpolate the cartesian
        coordinates of nearest neighbour molecules in the
        two parents
        """
        natoms = p0.shape[0]
        # coordinates of parents and list of first atom of each ligand
        child0 = np.copy(p0)
        child1 = np.copy(p1)
        # SBX parameters
        if self.SBX:
            coin = np.random.rand()
            if coin <= 0.5:
                beta = 2.*coin**(1./(self.eta + 1.))
            else:
                beta = (0.5*(1. - coin))**(1./(self.eta + 1.))
        # build children coordinates
        atoms1 = list(range(self.at1, natoms, self.nat))
        used = list()
        for at in atoms1:
            # search nearest at1 if not already used;
            ligands = list(set(atoms1).difference(used + [at]))
            if len(ligands) == 0:
                break
            atn = ligands[np.argmin(np.linalg.norm(child1[ligands]-child0[at],axis=1))]
            # check rmsd w/o rotation (centering only)
            # if below threshold interpolate
            l1 = p0[at:at + self.nat,:] - np.average(p0[at:at + self.nat,:], axis=0)
            ln = p1[atn:atn + self.nat,:] - np.average(p1[atn:atn + self.nat,:], axis=0)
            rmsd = quaternion_utils.rmsd(l1, ln)
            #if below rmsd interpolate            
            if rmsd <= self.rmsd_cutoff and np.linalg.norm(child1[atn] - child0[at]) <= self.ext_cutoff:
                if self.SBX:
                    child0[at:at+self.nat] = 0.5* ((1+beta)*p0[at:at+self.nat] + (1.-beta)*\
                        p1[atn:atn+self.nat])
                    child1[atn:atn+self.nat] = 0.5* ((1-beta)*p0[at:at+self.nat] + (1.+beta)*\
                        p1[atn:atn+self.nat])
                else:
                    child0[at:at+self.nat] = self.alpha*p0[at:at+self.nat] + (1.-self.alpha)*\
                        p1[atn:atn+self.nat]
                    child1[atn:atn+self.nat] = (1.-self.alpha)*p0[at:at+self.nat] + self.alpha*\
                       p1[atn:atn+self.nat]
                used.append(atn)
            else:
            #if not, try to swap
                child0[at:at+self.nat] = p1[at:at+self.nat]
                child1[at:at+self.nat] = p0[at:at+self.nat]
        # check for 
        # (i)  intramolecular collisions (cutoff_int)
        # (ii) that all first atoms in ligands are within cutoff_ext of starting posisions
        m0 = np.min(sp.spatial.distance.pdist(child0))
        m1 = np.min(sp.spatial.distance.pdist(child1))
        m2 = np.min(sp.spatial.distance.cdist(child0[self.at1::self.nat], child1[self.at1::self.nat]))
        if m0 <= self.int_cutoff or m1 <= self.int_cutoff or m2 >= self.ext_cutoff:
            interpolate_ok = False
        else:
            interpolate_ok = True
        
        if self.check(child0) or self.check(child1):
            interpolate_ok = False
        # if interpolation is ok return children 
        # otherwise swap parents
        if interpolate_ok == False:
            child0 = np.copy(p1)
            child1 = np.copy(p0)
        return child0, child1    
