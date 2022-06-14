import numpy   as np    
import os
import re
import scipy   as sp
from scipy.spatial.distance import pdist
import rotation

# Giordano Mancini Feb 2019

deg2rad = np.pi/180.
rad2deg = 1./deg2rad

def check_clash(coordinates,threshold):
    """
    check if minimum distance in coordinates
    is below threshold
    """
    D = pdist(coordinates)
    if np.min(D) <= threshold:
        return True
    else:
        return False

class reac_pair(object):
    """
    Container object that includes cartesian coordinates, an ID
    and attributes to manipulation of coordinates on fragment 2.
    Used to run MC simulation in which  fragment 2 (CD) rotates
    and translates towards the B end of fragment 1 (AB)
    Fragments are identified by atom intervals (2-tuples)
    i (AB),j,k (CD) are pivot atoms 
    NB ALL VALUES IN ANGSTROM
    - coordinates: np.ndarray natoms x 3 
    - ID to identify QM corresponding QM calculation
    The following methods are borrowed from ga_utils.specimen:
    - set_fitness
    - get_fitness
    - get_ID
    - set_ID
    """
    def __init__(self,template,AB=None,CD=None,ijkl=None,gen_trj=False):
        self.fitness = np.nan
        self.ID      = np.nan
        coords = re.compile(r'(!\sheader\ssection\send\n)(.*)()!\scoordinates\send',re.DOTALL)
        templ = open(template,"r")
        templL = templ.read()  
        records = (coords.search(templL).group(2)).split("\n")
        self.atoms = list()
        xyz = list()
        for i in records:
            if len(i) > 0:
                line = list(map(lambda x: x,i.split()))
                self.atoms.append(line[0])
                xyz.append(list(map(float,line[1:])))
        templ.close()
        self.natoms = len(self.atoms)
        self.XYZ = np.array(xyz)
        if check_clash(self.XYZ,0.6):
            raise ValueError("Clash in coordinates")
        #
        if (AB==None or CD==None) and gen_trj==False:
            raise ValueError("I need AB and CD")
        elif gen_trj:
            AB = list(range(0,self.natoms//2))
            CD = list(range(self.natoms//2,self.natoms+1))
        if ijkl==None and gen_trj==False:
            raise ValueError("I need pivots")
        elif gen_trj:
            ijkl = ((0,self.natoms//2,self.natoms))
        self.ijkl = np.array(ijkl)-1
        self.AB = list(range(AB[0]-1,AB[1]))
        self.CD = list(range(CD[0]-1,CD[1]))
              
    def set_energy(self,energy):
        self.energy = energy
        
    def get_energy(self):
        return self.energy

    def get_ID(self):
        return self.ID

    def set_ID(self,ID):
        if(ID > 0):
            self.ID = ID
        else:
            raise ValueError
        
    def set_coordinates(self, xyz):
        self.XYZ = xyz
                
    def get_coordinates(self):
        return self.XYZ
    
    def dump_coords(self):
        xyz = ""
        for i in range(self.XYZ.shape[0]):
            xyz = xyz + '{:4s} {:10.7f} {:10.7f} {:10.7f}\n'\
                .format(self.atoms[i],self.XYZ[i][0],self.XYZ[i][1],self.XYZ[i][2])
        return xyz
    
    def write_xyz(self):
        xyz = '{:5d} \n structure ID {:5f}\n'.format(self.natoms,self.ID)
        for i in range(self.XYZ.shape[0]):
            xyz = xyz + '{:4s} {:10.7f} {:10.7f} {:10.7f}\n'\
                .format(self.atoms[i],self.XYZ[i][0],self.XYZ[i][1],self.XYZ[i][2])
        return xyz    

### rotate and translate frag B

def gen_structure_1(reac_pair,cutoff=0.8,displ=0.05,ang=5.,ptrasl=0.5,maxtry=100):
    """
    generate new AB/CD configuration rotating and translating CD
    discard if CD is not directed towards AB or if there is a clash
    """
    coin = np.random.rand()
    ntry = 0
    dmin = 10.
    XYZ  = np.copy(reac_pair.XYZ)
    while True:
        u = rotation.unit_vector()
        Q = rotation.unit_quat(np.random.normal(0.,ang))
        for i in range(len(reac_pair.CD)):
            if coin < ptrasl:
                XYZ[reac_pair.CD[i]] += displ*u
            else:
                XYZ[reac_pair.CD[i]] = rotation.quat_action(Q,reac_pair.XYZ[reac_pair.CD[i]])
        dmin = np.min((dmin,np.min(np.linalg.norm(\
                XYZ[reac_pair.AB]- XYZ[reac_pair.CD[i]],axis=1))))
        d02 = np.linalg.norm(XYZ[reac_pair.ijkl[0]] - XYZ[reac_pair.ijkl[2]])
        d12 = np.linalg.norm(XYZ[reac_pair.ijkl[1]] - XYZ[reac_pair.ijkl[2]])
        d03 = np.linalg.norm(XYZ[reac_pair.ijkl[0]] - XYZ[reac_pair.ijkl[3]])
        d13 = np.linalg.norm(XYZ[reac_pair.ijkl[1]] - XYZ[reac_pair.ijkl[3]])
        check1 = (d02 < d03) and (d02 < d13)
        check2 = (d12 < d03) and (d12 < d13)
        if dmin > cutoff and check1 and check2:
            break
        elif ntry > maxtry:
            break
        else:
            ntry += 1
    if ntry <= maxtry:
        reac_pair.set_coordinates(XYZ)
    return reac_pair 

def gen_structure_2(reac_pair,cutoff=0.8,displ=0.1,ang=15.,ptrasl=0.5,maxtry=100):
    """
    generate new AB/CD configuration rotating and translating CD
    discard if CD is not directed towards AB or if there is a clash
    """
    coin = np.random.rand()
    ntry = 0
    dmin = 10.
    XYZ  = np.copy(reac_pair.XYZ)
    while True:
        u = rotation.unit_vector()
        Q = rotation.unit_quat(np.random.normal(0.,ang))
        for i in range(len(reac_pair.CD)):
            if coin < ptrasl:
                XYZ[reac_pair.CD[i]] += displ*u
            else:
                XYZ[reac_pair.CD[i]] = rotation.quat_action(Q,reac_pair.XYZ[reac_pair.CD[i]])
        dmin = np.min((dmin,np.min(np.linalg.norm(\
                XYZ[reac_pair.AB]- XYZ[reac_pair.CD[i]],axis=1))))
        d02 = np.linalg.norm(XYZ[reac_pair.ijkl[0]] - XYZ[reac_pair.ijkl[2]])
        d03 = np.linalg.norm(XYZ[reac_pair.ijkl[0]] - XYZ[reac_pair.ijkl[3]])
        d12 = np.linalg.norm(XYZ[reac_pair.ijkl[1]] - XYZ[reac_pair.ijkl[2]])
        d13 = np.linalg.norm(XYZ[reac_pair.ijkl[1]] - XYZ[reac_pair.ijkl[3]])
        check1 = d02 < d03 
        check2 = d12 < d13
        if dmin > cutoff and check1 and check2:
            break
        elif ntry > maxtry:
            break
        else:
            ntry += 1
    if ntry <= maxtry:
        reac_pair.set_coordinates(XYZ)
    return reac_pair 
