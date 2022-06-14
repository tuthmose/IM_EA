import numpy as np    
import os
import re
import scipy as sp
import subprocess as sub 

import cluster_utils
import sa_reacpair
import zmatrix

# Giordano Mancini Dec 2018 / may 2019

def input_style():
    """
    regexps for input file: connectivity, complete header, other coordinates and dihedrals
    """
    REGX = dict()
    REGX['route']  = re.compile(r'(.*)(!\sroute\ssection\send\n)',re.DOTALL)
    REGX['header'] = re.compile(r'(!\sroute\ssection\send\n)(.*)(!\sheader\ssection\send\n)',re.DOTALL)
    REGX['coords'] = re.compile(r'(!\sheader\ssection\send\n)(.*)(!\scoordinates\send)',re.DOTALL)
    REGX['basis'] = re.compile(r'(!\scoordinates\send\n)(.*)(!\stemplate\send)',re.DOTALL)
    REGX['islink1'] = re.compile(r'(--Link1--)')
    REGX['amber'] = re.compile(r'.*(amber).*',re.IGNORECASE)
    REGX['uff'] = re.compile(r'.*(uff).*',re.IGNORECASE)
    REGX['alphac'] = re.compile(r'(^[0-9A-Za-z]+)((-.*|))')
    REGX['oniom'] = re.compile(r'(^[HL])')
    return REGX
              
def output_style(style):
    """
    regexps for output G16 file: final (zmat) coordinates, energy and optimization done
    """
    REGX = dict()
    REGX['opt'] = re.compile('Berny optimization')
    REGX['Error'] = re.compile('Error termination via Lnk1e')
    REGX['dipole'] = re.compile(r'(Tot=\s+\d+\.\d+)')
    REGX['fgeom'] = re.compile\
        (r'(Optimization\scompleted.*Input\sorientation:.*Z\s+-+\s+)(.*)(\s-+\n)(\s+(Distance|Rotational).*\s)',\
            re.DOTALL)
    REGX['fgeom_sp'] = re.compile\
            (r'Input\sorientation:.*\s\-+\n(.*)\s\-+\n\s+Distance\smat',re.DOTALL)
    if style == "DFT":
        REGX['lastE'] = re.compile(r'SCF Done:\s+E\(.*\)\s=\s+((\s|-)\d+\.\d+(E-\d+|\s))\s')
        REGX['CnvOk'] = re.compile('Optimization completed')
        REGX['Stopped'] = re.compile('Optimization stopped')
        REGX['fgeom_l1'] =\
            re.compile\
                (r'(Enter.*l202\.exe.*Input\sorientation:.*Z\s+-+\s+)(.*)(\s-+\n)(\s+Distance.*\s)',\
                    re.DOTALL)       
    elif style == "MM":
        REGX['lastE'] = re.compile(r'(\sEnergy=\s+.*)(NIter.*\n)(\sDipole\smoment=)')
        REGX['CnvOk'] = re.compile('Optimization completed')
        REGX['Stopped'] = re.compile('Optimization stopped')
    return REGX

class gaucaller:
    """
    wrapper object to calculate fitness function using gaussian in a subprocess
    the template must be the same with which chromosomes are created
    internal coordinates (zmatrix instance attributes )are changed everytime it is invoked
    """
    def __init__(self,**kwargs):
        """
        - template is a Gaussian input file used as template (never modified)
        - ID is the number of invocations to the caller to generate unique file names
        - filename is a filename prefix
        - shellname is a tcsh script used to execute gaussian
        - notconv assigns a default fitness value to non converged calculations
        - stopok determines is energies from non converged optimizations are ok
        - sptype specifies that we are optimizing a molecule (mol) or a cluster (clust)
        - zm_in  writes a zm input files instead of cartesian coordinates
        - run_alw use try/except block in call to gaussian
        Internal coordinates with zmatrix are available only for molecules
        """
        # init some variables
        prop_defaults = {
            "ID"         : 0,
            "stopok"      : False,
            "filename"    : None,
            "shellname"   : None,
            "template"    : None,
            "notconv"     : 100.,
            "optstr"      : 'opt ',
            "sptype"      : "mol",
            "scratch"     : "./",
            "zm_in"       : False,
            "run_alw"     : False,
            "cutoff"      : 0.7,
            "bdelta"      : 0.2,
            "check_bonds" : False,
            "run_esc"     : True
            }
        for (prop, default) in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))      
        if self.shellname is None:
            raise ValueError("tcsh shell for g16 not provided")
        if self.filename is None:
            raise ValueError("File name prefix for creating files not provided")
        if self.template is None:
            raise ValueError("G16 template not provided")
        assert self.sptype in ("mol","clust")
        if self.run_esc is False:
            print("Dry run: no ESC calculations")
        # regexps for input file: connectivity, complete header, other coordinates and dihedrals
        self.REGX = input_style()
        
        # open and parse template file
        templ = open(self.template,"r")
        templL = templ.read()  
        route = self.REGX['route'].search(templL).group(1)
        self.Route = route.rstrip("\r\n")+"\n"
        self.Hdr = self.REGX['header'].search(templL).group(2)
        # load cartesian coordinates for cluster
        if self.sptype == "clust":
            records = (self.REGX['coords'].search(templL).group(2)).split("\n")
            initX   = list()
            for i in records:
                I = i.split()
                if len(I) == 0:
                    continue
                elif len(I) == 4:
                    line = list(map(lambda x: float(x), I[1:]))
                elif len(I) == 5 and (I[-1]=='H' or I[-1]=='L'):
                    line = list(map(lambda x: float(x), I[1:-1]))
                elif len(I) == 5:
                    line = list(map(lambda x: float(x), I[2:]))
                elif len(I) == 6:
                    line = list(map(lambda x: float(x), I[2:-1]))
                initX.append(line)
            initX = np.asarray(initX)
            #check for OptAt column
            if initX.shape[1] > 3:
                self.initX = initX[:,-3:]
            else:
                self.initX = initX
        #load any additional input
        add_input = self.REGX['basis'].search(templL)
        if add_input is not None:
            self.add_input = add_input.group(2)
        else:
            self.add_input = None
        Link1 = self.REGX['islink1'].search(templL)
        if Link1 != None:
            self.Link1 = True
        else:
            self.Link1 = False
        templ.close()
        
        #check if is a MM calculation
        MMamber = self.REGX['amber'].search(self.Route)
        MMuff   = self.REGX['uff'].search(self.Route)
        if MMuff is not None or MMamber is not None:
            self.style = "MM"
        else:
            self.style = "DFT"
        #set log file regexpes
        self.OREGX = output_style(self.style)         
    
    def writeinp(self, coords, com_name, flex):
        """
        given coordinates writes input file
        - flex: wether to run a geometry optimization
        - optstr: additional options for optimization
        """
        com_file = open(com_name,"w")
        com_file.write(self.Route)
        if flex:
            com_file.write(self.optstr)
            com_file.write("\n")
        com_file.write("! route section end\n")
        #com_file.write('\n')
        com_file.write(self.Hdr)
        com_file.write("! header section end\n")
        com_file.write(coords)
        com_file.write("! coordinates end\n")
        if self.add_input is not None:
            com_file.write(self.add_input)
        else:
            com_file.write('\n')
        com_file.close()
        return None
    
    def call_esc(self,com_name,log_name,namedir,structure,flex):
        """
        call G16
        """
        ### call gaussian
        args = "./"+self.shellname+" "+com_name+" "+log_name+" "+namedir
        #args = self.shellname+" "+com_name+" "+log_name+" "+namedir
        if self.run_alw:
            try:
                sub.run(args, stderr=sub.PIPE, shell=True, cwd=os.getcwd())
            except:
                print("Error running run ",self.ID, log_name)
                return self.notconv, structure.get_coordinates()
        else:
            sub.run(args, stderr=sub.PIPE, shell=True, cwd=os.getcwd())
            
        ### parse output: check for error or convergence or get en
        Ok   = None
        Stop = None
        Err  = None
        log_file = open(log_name,"r")
        log  = log_file.read()
        if flex is False:
            Ok   = "Ok"
        else:
            Ok = self.OREGX['CnvOk'].search(log)
        Stop = self.OREGX['Stopped'].search(log)
        Err = self.OREGX['Error'].search(log)
        energy = self.notconv
        if (Err is not None) or ((Ok is None) and (Stop is not None and self.stopok is False)):
            # run was terminated by error or because of convergence, do not return anything
            energy = self.notconv
        elif (Ok is not None) or (Ok is None and (Stop is not None and self.stopok is True)):
            if self.style == "DFT":
                energy = self.OREGX['lastE'].findall(log).pop()
            elif self.style == "MM":
                try:
                    energy = self.OREGX['lastE'].search(log).group(1)
                    energy = energy.split()[1]
                except:
                    print("energy not found in ",log_name)
                    raise ValueError
            if isinstance(energy,tuple):
                energy = float(energy[0])
            else:
                energy = float(energy)             
            if flex:
                if self.Link1:
                    X = self.OREGX['fgeom_l1'].search(log).group(2)
                else:
                    X = self.OREGX['fgeom'].search(log).group(2)
                data = list(map(lambda x: x.split(),X.splitlines()))
                cartesian = list()
                for d in data:
                    cartesian.append(d[3:])
                # even if an energy value is loaded actual coordinates may still make no sense
                # e.g. they have values like 1000.
                try:
                    cartesian = np.asarray(cartesian,dtype='float')
                except:
                    energy = self.notconv
                #check for clashes
                if energy != self.notconv:
                    Dist = sp.spatial.distance.pdist(cartesian)
                    if Dist.min() <= self.cutoff: 
                        energy = self.notconv
                    else:
                       #optionally, check for broken bonds
                        if self.check_bonds:
                            broken = structure.zmat.check_broken_bonds(cartesian,self.bdelta)
                        else:
                            broken = False
                        if broken: 
                            energy = self.notconv
                        elif self.sptype == "mol":
                            structure.zmat.update_internal_coords(cartesian)
                        else:
                            structure.set_coordinates(cartesian)
        structure.set_energy(energy)
        return energy

    def fitcalc(self, structure, **kwargs):
    #def fitcalc(self,structure,flex=None):
        """
        Fitness calculator:
            - called on a specimen or cluster instance
            - creates gaussian input files with modified zmat
            - calls G16 or just generates output structure and 
              returns a random number ("fake fitness")
            - parses log file and get energy and optimized geometry
            - filename created from ID
        for molecules and internal moves:
            structure is a molecular system created with ga_utils, zmatrix 
        for clusters: 
            structure is cluster created with cluster_utils
        fitness and dihedrals or coordinates are returned
        geometry is altered in place
        """
        ### init file
        self.ID += 1
        if self.ID < 10:
            xnum = "0000"
        elif self.ID >= 10 and self.ID < 100:
            xnum = "000"
        elif self.ID >= 100 and self.ID < 1000:
            xnum = "00"
        elif self.ID >= 1000 and self.ID < 10000:
            xnum = "0"
        else:
            print(self.ID)
            raise ValueError("GauParser: too many input files")
        com_name = self.filename + "_" + str(xnum) + str(self.ID) + ".com"
        log_name = self.filename + "_" + str(xnum) + str(self.ID) + ".log"
        
        ### write values in input file
        flex = kwargs['flex']
        if self.sptype == "mol":
            ZM = structure.zmat.dump_coords(retc=True,zstr=True)
        elif self.zm_in is True:
            ZM = structure.zmat.write_zmat(False)
        elif self.sptype == "clust":
            ZM = structure.dump_coords()
        if self.run_esc:
            self.writeinp(ZM, com_name, flex)
            namedir = self.scratch + "dir_"+self.filename+str(self.ID)
        #set ID
        structure.set_ID(self.ID)
        
        if self.sptype == "mol":
            cl = structure.zmat.check_clash(self.cutoff)
            if cl:
                print("cl",self.ID,structure.get_coordinates())
                raise ValueError("Detected clash in generating input file")

        # call esc or just return geom
        if self.run_esc:
            energy = self.call_esc(com_name, log_name, namedir, structure, flex)
        else:
            energy = np.random.rand()
            log_name = self.filename + "_" + str(xnum) + str(self.ID) + ".xyz"
            if self.sptype == "clust":
                atoms = structure.atoms
                xyz = structure.get_coordinates()
                cluster_utils.write_xyz(atoms, xyz, log_name)
            else:
                structure.zmat.write_file(log_name, "xyz")

        #return here
        return energy, structure.get_coordinates()
