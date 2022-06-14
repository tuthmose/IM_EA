from copy import deepcopy as deepc
from scipy.stats import circmean
from scipy.spatial.distance import pdist,squareform
import math
import numpy as np
import re
import scipy as sp
import string

# Giordano Mancini Dec 2018

deg2rad = np.pi/180.
rad2deg = 1./deg2rad
circle  = 2.*np.pi
    
def wrap(angle):
    if angle > circle:
        return angle % circle
    elif angle < 0. :
        return angle % -circle
    else:
        return angle    

def deg_circ_mean(angles):
    alpha = deg2rad*np.asarray(angles)
    theta = circmean(alpha)
    return rad2deg*theta

def deg_circ_dist(alpha,beta):
    delta = alpha - beta
    delta = (delta + 180.) % 360. - 180.
    return delta
    
class zmat:
    def __init__(self,template,rot_bonds=None,fmt="gau"):
        """
        create initial instance of zmat converter 
        and parse template zmat file
        rot_bonds is a tuple of 2-tuple of rotatable
        bonds used for generating dihedral moves
        """
        #create data
        self.bonds  = dict()
        self.angles = dict()
        self.dih    = dict()
        self.atom1  = dict()
        self.atom2  = dict()
        self.atom3  = dict()
        self.Atoms  = list()
        if rot_bonds is not None:
            self.rot_bonds = list()
            for bond in rot_bonds:
                self.rot_bonds.append((bond[0]-1,bond[1]-1))
            self.nrot = len(self.rot_bonds)
        else:
            self.rot_bonds = None

        #parse template
        zmat = open(template,"r")
        if fmt == "gau":
            self.read_gau(zmat)
        elif fmt == "zmat":
            self.read_zm(zmat)
        
        #check that zmat is coherent with r. bonds and fill adiacency matrix
        self.natoms = len(self.Atoms)
        self.admat = np.zeros((self.natoms,self.natoms),dtype='int')
        if self.rot_bonds is not None:        
            all_bonds = list()
            for i in range(self.natoms):
                if i in self.atom1 and i in self.atom2:
                    all_bonds.append((self.atom1[i],self.atom2[i]))
                    self.admat[self.atom1[i],self.atom2[i]] =\
                    self.admat[self.atom1[i],self.atom2[i]] = 1
            for i in self.rot_bonds:
                j = (i[1],i[0])
                if i not in all_bonds and j not in all_bonds:
                    raise ValueError("{0:6d}{1:6d} not in bonds".\
                                 format(i[0]+1,i[1]+1))
        #update cartesian coordinates
        self.xyz = self.calc_coords(None,None)
        self.bonded_pairs = tuple(self.bonds.items())

    def read_gau(self,zmat):
        """
        read gaussian input file with zmatrix
        values in mol. spec (var=line) or separate
        list (var=sym)
        """
        zmat_records = zmat.read()
        rgx1 = re.compile(r'!\svariable\slist')
        var = rgx1.search(zmat_records)
        if var != None:
            #load var values first
            vlist = re.compile(r'(.*)(!\svariable\slist\n)(.*)(!\scoordinates\send\n)',re.DOTALL)
            records = vlist.search(zmat_records).group(3)
            records = records.split("\n")
            VARS = dict()            
            for v in records:
                if len(v) > 0:
                    line = v.split()
                    VARS[line[0]] = float(line[1])
            header   = re.compile(r'(.*)(!\sheader\ssection\send\n)(.*)(!\svariable\slist\n)',re.DOTALL)
        else:
            header = re.compile(r'(.*)(!\sheader\ssection\send\n)(.*)(!\scoordinates\send\n)',re.DOTALL)
        #now read topology
        records = header.search(zmat_records).group(3)
        records = records.split("\n")            
        for l,record in enumerate(records):
            line = record.split()
            L = len(line)
            if L==0: continue
            self.Atoms.append(line[0])
            if L>1:
                self.atom1[l] = int(line[1])-1
                if var==None:
                    self.bonds[l] = float(line[2])
                else:
                    self.bonds[l] = VARS[line[2]] 
            if L > 3:
                self.atom2[l] = int(line[3])-1
                if var==None:
                    self.angles[l] = float(line[4])*deg2rad
                else:
                    self.angles[l] = VARS[line[4]]*deg2rad
            if L > 5:
                self.atom3[l] = int(line[5])-1
                if var==None:
                    self.dih[l] = float(line[6])*deg2rad
                else:
                    self.dih[l] = VARS[line[6]]*deg2rad            
        
    def read_zm(self,zmat):
        """
        read zmatrix file from molden, gv or stuff like that
        """
        zmat_records = zmat.readlines()
        natoms = int(zmat_records[0])
        offset = 1
        if len(zmat_records) > natoms+2:
            varlist = True
        else:
            varlist = False
        if varlist:
            VARS = dict()
            for l in range(natoms+3,len(zmat_records)):
                line = zmat_records[l].split()
                try:
                    VARS[str.rstrip(line[0].replace("=",""))] = float(line[1])
                except:
                    break
            print(VARS)
            for ll in range(1, natoms+1):
                l = ll-offset
                line = zmat_records[ll].split()
                L = len(line)
                if L==0: 
                    continue
                self.Atoms.append(line[0])
                if L>1:
                    self.atom1[l] = int(line[1]) - 1
                    self.bonds[l] = VARS[line[2]] 
                if L > 3:
                    self.atom2[l]   = int(line[3]) - 1
                    self.angles [l] = VARS[line[4]]*deg2rad
                if L > 5:
                    self.atom3[l] = int(line[5]) - 1
                    self.dih[l]   = VARS[line[6]]*deg2rad         
            print(self.dih)
        else:
            for ll in range(2, natoms+2):
                l = ll-offset
                line = zmat_records[ll].split()                
                # line = record.split()
                L = len(line)
                if L == 0:
                    continue
                self.Atoms.append(line[0])
                if L>1:
                    self.atom1[l] = int(line[1]) - 1
                    self.bonds[l] = float(line[2])
                if L > 3:
                    self.atom2[l] = int(line[3]) - 1
                    self.angles [l] = float(line[4])*deg2rad
                if L > 5:
                    self.atom3[l] = int(line[5]) - 1
                    self.dih[l] = float(line[6])*deg2rad            
            zmat.close()
            
    def rotate(self,dihedrals,theta,bond_num):
        rble_bond = self.rot_bonds[bond_num]
        first_done = False
        ratom = None
        for atom in range(self.natoms):
            if atom in self.atom3:
                r1 = (self.atom1[atom],self.atom2[atom])
                r2 = (self.atom2[atom],self.atom1[atom])   
                r3 = (self.atom3[atom],self.atom2[atom])
                r4 = (self.atom2[atom],self.atom3[atom])
                if rble_bond==r1 or rble_bond==r2:
                    if first_done == False:
                        dihedrals[atom] = theta
                        delta = dihedrals[atom] - self.dih[atom]
                        first_done = True
                        ratom = atom
                    elif self.atom3[atom] != ratom:
                        dihedrals[atom] = wrap(self.dih[atom] + delta)
                elif self.atom1[atom] == ratom and first_done and (rble_bond==r3 or rble_bond==r4):
                        dihedrals[atom] = wrap(self.dih[atom] + delta)
        
    def update_dihedrals(self,newdih,select,update=False):
        """
        rotate about rotatable bonds checking
        for rotations depending on the same 
        atoms involved in the input
        """
        newdih = np.asarray(newdih)*deg2rad
        assert len(newdih) == len(select)
        dihedrals = self.dih.copy()
        for ang, bond_num in enumerate(select):
            self.rotate(dihedrals,newdih[ang],bond_num)
        if update is True:
            self.dih = dihedrals
            self.xyz = self.calc_coords(None,None)
        return dihedrals
    
    def update_clash(self,newdih,select,cutoff):
        """
        try to update a single dihedral
        and check for clash;
        if there is a clash return False
        and restore dihedrals otherwise update
        if there is no clash and check_bonds is
        True check also that no bonds were elongated
        by more than a given threshold
        """
        newdih = newdih[0]*deg2rad
        rble_bond = select[0]
        dihedrals = self.dih.copy()
        olddih    = self.dih.copy()
        self.rotate(dihedrals,newdih,rble_bond)
        #note that dihedrals is changed in place by rotate
        self.dih = dihedrals
        clash = self.check_clash(cutoff,calc=True)
        if clash:
            self.dih = olddih.copy()
            self.xyz = self.calc_coords(None,None)            
            return True
        else:
            return False
                                
    def mindist(self,calc):
        if calc:
            self.xyz = self.calc_coords(None,None)
        S = sp.spatial.distance.pdist(self.xyz)
        return S.min()       
        
    def check_clash(self,cutoff,calc=True):
        """
        check if given coordinates create collisions
        within given cutoff
        """
        D = self.mindist(calc)
        if D <= cutoff:
            return True
        else:
            return False
    
    def update_internal_coords(self,xyz):
        """
        given a set of cartesian coordinates,
        update zmatrix
        """
        self.xyz = xyz
        for i in range(1,self.natoms):
            v1 = xyz[i] - xyz[self.atom1[i]]
            self.bonds[i] = np.linalg.norm(v1)
            if i == 1: 
                continue
            v2 = xyz[self.atom2[i]] - xyz[self.atom1[i]]
            theta = math.atan2(np.linalg.norm(np.cross(v1,v2)),np.dot(v1,v2))
            self.angles[i] = theta
            if i == 2: 
                continue
            v2 = -v2
            v3 = -xyz[self.atom3[i]] + xyz[self.atom2[i]] 
            # normal to first semiplane (atom, atom1, atom2)
            b1 = np.cross(v1,v2)
            # normal to second semiplane (atom1, atom2, atom3)
            b2 = np.cross(v2,v3)
            n1 = np.linalg.norm(b1)
            n2 = np.linalg.norm(b2)
            b1 = b1/n1
            b2 = b2/n2
            v2 = v2/np.linalg.norm(v2)
            # normal to semiplane between n1 and atom1,atom2,atom3
            m1 = np.cross(b1,v2)
            phi = math.atan2(np.dot(m1,b2),np.dot(b1,b2))            
            #if phi < 0: phi += 2.*np.pi
            self.dih[i] = phi
        
    def calc_coords(self,newdih=None,select=None):
        """
        Self Normalizing Natural Extension 
        Reference Frame (SN Nerf) algorithm
        see 10.1002/jcc.20237
        - newdih is a tuple of floats with new dihedrals
        - select is a tuple of 2-tuples with rotatable bonds to
          be changed
        other atoms on the same rotatable bond are rotated
        by [new angle] - [old angle] calculated for selected
        atoms
        """
        # empty new relative positions
        relPos = list()
        cart   = np.zeros((self.natoms,3))
        if self.natoms == 1:
            return cart
        cart[1,0] = self.bonds[1]
        if self.natoms == 2:
            return cart
        firstAt = 2
        
        # for more than 2 atoms, check if new dihedrals
        # have been assigned
        if newdih is None and select is not None:
            raise ValueError("selected rotating atoms but no new dihedrals")
        elif newdih is not None and select is None:
            raise ValueError("selected new dihedrals but no rotating atoms")
        elif newdih is not None and select is not None:
            dihedrals = self.update_dihedrals(newdih,select)
        else:
            dihedrals = self.dih
            
        #apply nerf algorithm
        for i in range(firstAt,self.natoms):
            r = self.bonds[i]
            cosTheta = math.cos(np.pi - self.angles[i])
            sinTheta = math.sin(np.pi - self.angles[i])
            if i > 2:
                cosPhi = math.cos(dihedrals[i])
                sinPhi = math.sin(dihedrals[i])
            else:
                cosPhi = 1.
                sinPhi = 0.
            relPos.append([r*cosTheta, r*cosPhi*sinTheta, r*sinPhi*sinTheta])
        
        for i in range(firstAt,self.natoms):
            RotMat = np.zeros((4,4))
            b_to_c = cart[self.atom1[i]] - cart[self.atom2[i]]
            b_to_c = b_to_c / np.linalg.norm(b_to_c)
            if i == 2:
                a_to_b = np.array((0.,-1.,0.))
            else:
                a_to_b = cart[self.atom2[i]] - cart[self.atom3[i]]
                a_to_b = a_to_b / np.linalg.norm(a_to_b)
            n   = np.cross(a_to_b,b_to_c)
            n   = n/np.linalg.norm(n)
            nbc = np.cross(n,b_to_c)
            
            RotMat[0,0] = b_to_c[0]
            RotMat[1,0] = b_to_c[1]
            RotMat[2,0] = b_to_c[2]
            #RotMat[3,0] = 0
            RotMat[0,1] = nbc[0]
            RotMat[1,1] = nbc[1]
            RotMat[2,1] = nbc[2]
            #RotMat[3,1] = 0
            RotMat[0,2] = n[0]
            RotMat[1,2] = n[1]
            RotMat[2,2] = n[2]
            #RotMat[3,2] = 0                
            RotMat[0,3] = cart[self.atom1[i],0]
            RotMat[1,3] = cart[self.atom1[i],1]
            RotMat[2,3] = cart[self.atom1[i],2]
            RotMat[3,3] = 1
            tmp = np.append(relPos[i-firstAt],0)
            cart[i] = np.dot(RotMat,tmp)[:3]+cart[self.atom1[i]]
        return cart
    
    def check_broken_bonds(self,xyz,bdelta):
        """
        given new coords xyz, still to be saved,
        check if bonds are elongated by more
        than XXX or if non bonded atoms are within
        0.9 angs unless they previously were
        """
        #for bond in self.bonds:
        #    r0 = np.linalg.norm(self.xyz[bond]-self.xyz[self.bonds[bond]])
        #    r1 = np.linalg.norm(xyz[bond]-xyz[self.bonds[bond]])
        #    if abs(r1-r0) > bdelta:
        #        return True
        #        break
        D = squareform(pdist(xyz))
        Dold = squareform(pdist(self.xyz))
        Dlow = np.where(D<=1.0)
        for a,A in enumerate(Dlow[0]):
            B = Dlow[1][a]
            if A==B:
                continue
            elif not (A,B) in self.bonded_pairs:
                return True
            elif Dold[A,B]> 1.0:
                return True
        return False
        
    def write_zmat(self,watoms=True):
        """
        return a string with zmatrix in symbolic form
        assumes dihedrals have been updated if needed
        """
        ZM = ""
        nbonds  = 0
        nangles = 0
        B = dict()
        A = dict()
        if watoms:
            ZM = ZM + str(self.natoms) + "\n\n"
        for i, atom in enumerate(self.Atoms):
            ZM = ZM + "{0:6s}".format(atom)
            if i in self.atom1:
                nbonds += 1
                b = "r" + str(nbonds)
                ZM = ZM + "{0:6d}  {1:s}".format(self.atom1[i]+1,b)
                B[nbonds] = self.bonds[i]
            if i in self.atom2:
                nangles += 1
                a = "a" + str(nangles)
                ZM = ZM + "{0:6d}  {1:s}".format(self.atom2[i]+1,a)
                A[nangles] = self.angles[i]*rad2deg
            if i in self.atom3:
                ZM = ZM + "{0:6d}  {1:8.4f}".format(self.atom3[i]+1,self.dih[i]*rad2deg)
            ZM = ZM + "\n"
        ZM = ZM + "\n"
        for i in range(1,nbonds+1):
            S = "{0:6d}= {1:9.6f}\n".format(i,B[i])
            S = "r" + str.lstrip(S)
            ZM = ZM + S
        for i in range(1,nangles+1):
            S = "{0:6d}= {1:8.4f}\n".format(i,A[i])
            S = "a" + str.lstrip(S)
            ZM = ZM + S
        return ZM
        
    def dump_coords(self,newdih=None,select=None,retc=False,prt=False,zstr=False, xtb=False):
        """
        print cartesian coordinates on stdout and/or return them as array
        or string
        """
        if newdih is None or select is None:
            coords = self.xyz
        else:
            coords = self.calc_coords(newdih,select) 
        if prt is True:
            for i in range(coords.shape[0]):
                print('{:4s} {:10.7f} {:10.7f} {:10.7f}'.\
                    format(self.Atoms[i],coords[i,0],coords[i,1],coords[i,2]))
        if xtb:
            xyz = ""
            for i in range(coords.shape[0]):
                xyz = xyz + '{a[0]:10.7f} {a[1]:10.7f} {a[2]:10.7f} {b:4s}\n'.format(a=coords[i], b=self.Atoms[i])
            return xyz
        if retc is True:
            if zstr is True:
                myxyz = ""
                for i in range(coords.shape[0]):
                    myxyz = myxyz + '{:4s} {:10.7f} {:10.7f} {:10.7f}\n'.\
                        format(self.Atoms[i],coords[i,0],coords[i,1],coords[i,2])
                return myxyz
            else:
                return coords
            
    def write_file(self,name,fmt,template=None,newdih=None,select=None):
        """
        write .gro or .pdb coordinate file with new dihedrals applied to rotatable bonds
        name is the template file with atom, residues and other non coordinate info
        """     
        if newdih is None or select is None:
            coords = self.xyz
        else:
            coords = self.calc_coords(newdih,select)
        if fmt == "gro":
            coords = 0.1 * coords
        elif fmt == "pdb":
            pass
        elif fmt == "xyz":
            pass
        else:
            raise ValueError("Output format not supported")
        conf = open(name,"w")          
        if fmt != "xyz":
            template_file = open(template,"r")
            tpl_lines = template_file.readlines()

        #header
        if fmt == "pdb":
            conf.write("MODEL\n")
            conf.write(tpl_lines[0])
        elif fmt == "gro":
            conf.write(tpl_lines[0])
            conf.write(tpl_lines[1])
        else:
            conf.write(str(self.natoms)+"\n\n")
            
        for l in range(len(coords)):
            if fmt == "gro": 
                record = tpl_lines[l+2][:-1]+"{0:7.3f}".\
                format(coords[l,0])+"{0:8.3f}".format(coords[l,1])\
                +"{0:8.3f}".format(coords[l,2])  +"\n"
            elif fmt == "pdb":
                record = tpl_lines[l+1][:-1]+"{0:9.3f}".\
                format(coords[l,0])+"{0:8.3f}".format(coords[l,1])\
                +"{0:8.3f}".format(coords[l,2])  + \
                "{0:6.2f}{1:6.2f}{2:>12s}\n".format(0.,0.,self.Atoms[l])
            elif fmt == "xyz":
                record = '{:4s} {:10.7f} {:10.7f} {:10.7f}\n'.\
                format(self.Atoms[l],coords[l,0],coords[l,1],coords[l,2])
            conf.write(record)
        if fmt == "gro": 
            conf.write(tpl_lines[-1])
        elif fmt == "pdb":
            conf.write("TER\nENDMDL\n")
        conf.close()
        if fmt=="gro" or fmt=="pdb":
            template_file.close()

