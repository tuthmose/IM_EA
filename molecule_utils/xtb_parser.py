import numpy as np    
import os
import re
import scipy as sp
import scipy.constants
import subprocess as sub 
import sys

import cluster_utils
import zmatrix

# Giordano Mancini Nov 2020

bohr2ang = scipy.constants.physical_constants['Bohr radius'][0]*10e9
ang2bohr = 1./bohr2ang

def input_style():
    REGX = dict()
    REGX['cmd_line'] = re.compile(r'(!\scmd\sline\sstart\n)(.*)(!\scmd\sline\send\n)',re.DOTALL)
    REGX['constraints'] = re.compile(r'(!\scmd\sline\send\n!)(.*)(\$coord\sangs\n)',re.DOTALL)
    REGX['coords'] = re.compile(r'\$coord\sangs\n(.*)\n\$end\n!\scoordinates\send',re.DOTALL)
    REGX['add_input'] = re.compile(r'!\scoordinates\send\n(.*)!\stemplate\send',
                                  re.DOTALL)
    return REGX
           
def output_style():
    REGX = dict()
    REGX['singlepE'] = re.compile(r'SUMMARY\s+::\n\s+:+\n\s+::\stotal\senergy\s+([\-0-9\.]+)\sEh')
    REGX['CnvOk'] = re.compile(r'GEOMETRY OPTIMIZATION CONVERGED')
    REGX['lastE'] = \
    re.compile(r'Final\sSinglepoint(.*)SUMMARY(.*)total\senergy\s+([\-0-9\.]+)\s+Eh',re.DOTALL)
    ## TODO find opt stopped msg
    REGX['Stopped'] = None
    REGX['fgeom'] = re.compile(r'\sfinal\sstructure:\n=+\n\$coord\n(.*)(\n\$.*)\$end',re.DOTALL)
    # this is applied to the coord file not the std out
    REGX['fgeom_sp'] = re.compile(r'\$coord\sangs\n(.*?)\n\$end',re.DOTALL)
    REGX['Error'] = re.compile('abnormal\stermination\sof\sxtb')
    REGX['dipole'] = re.compile('molecular\sdipole.*full([\-0-9\.]+)molecular\squadrupole')
    REGX['dpall'] = re.compile('molecular\sdipole:.*\n.*\n.*\n.*full:\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\n')
    REGX['failed'] = re.compile('FAILED\sTO\sCONVERGE\sGEOMETRY')
    return REGX

class xTBcaller:
    """
    wrapper object to calculate fitness function using Grimme's XTB 
    (version 6.3.3 (5b13467) currently)
    the template must be the same with which chromosomes are created
    """
    
    def __init__(self, **kwargs):
        """
        - template is a xTB/Turbomole input file used as template (never modified)
        - ID is the number of invocations to the caller to generate unique file names
        - filename is a filename prefix
        - notconv assigns a default fitness value to non converged calculations
        - stopok determines is energies from non converged optimizations are ok
        - sptype specifies that we are optimizing a molecule (mol) or a cluster (clust)
        - zm_in  writes a zm input files instead of cartesian coordinates
        - run_alw use try/except block in call to gaussian
          currently only for cartesian coordinates
        - charge, umpaired el. and other information is read in comment lines before
          the Turbomole input and passed to xtb with the exception of namespace
          which is given by filename
        """
        # init some variables
        prop_defaults = {
            "ID"         : 0,
            "stopok"      : False,
            "filename"    : None,
            "template"    : None,
            "notconv"     : 100.,
            "sptype"      : "clust",
            "scratch"     : "./",
            "zm_in"       : False,
            "cutoff"      : 0.9,
            "run_esc"     : True
            }
        for (prop, default) in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))
        if self.filename is None:
            raise ValueError("File name prefix for creating files not provided")
        if self.template is None:
            raise ValueError("xTB template not provided")
        ### TODO: change when you know how to use ICs in XTB
        # assert self.sptype == "clust"
        if self.run_esc is False:
            print("Dry run: no ESC calculations")
        # regexps for input file: connectivity, complete header, other coordinates and dihedrals
        self.REGX = input_style()

        # open and parse template file
        templ = open(self.template,"r")
        templL = templ.read()
        try:
            self.cmd_line = self.REGX['cmd_line'].search(templL).group(2)
        except:
            print(self.REGX['cmd_line'].search(templL))
            raise ValueError("Missing command line for xTB")
        try:
            self.constraints = self.REGX['constraints'].search(templL).group(2)
        except:
            self.constraints = None

        # load cartesian coordinates for cluster
        if self.sptype == "clust":
            records = (self.REGX['coords'].search(templL).group(1)).split("\n")
            initX   = list()
            for rec in records:
                line = rec.split()
                initX.append(line[:-1])
            self.initX = np.asarray(initX,dtype='float')

        # load additional information
        if self.REGX['add_input'].search(templL) is not None:
            self.add_input = self.REGX['add_input'].search(templL).group(1)
        else:
            self.add_input = None
        # close template
        templ.close()

        # set regexps to parse outputs
        self.OREGX = output_style()
            
    def writeinp(self, coords, coord_name):
        """
        given coordinates writes input file
        """
        coord_file = open(coord_name,"w")
        if self.constraints is not None:
            coord_file.write(self.constraints)
        coord_file.write("$coord angs\n")
        coord_file.write(coords)
        coord_file.write("$end\n")
        if self.add_input is not None:
            coord_file.write(self.add_input)
        coord_file.close()

    def call_esc(self, xtb_namespace, coord_name, out_name, structure, flex):
        """
        call xTB
        """
        args="./xtb_shell.sh ./" + coord_name + " " + self.cmd_line[2:-1] +\
            " --namespace " + xtb_namespace + " " + flex + " " + out_name
        sub.run(args, stderr=sys.stdout.buffer, shell=True, cwd=os.getcwd(), executable='/bin/bash')
        ### parse output: check for error or convergence or get en
        Ok   = None
        Stop = None
        Err  = None
        out_file = open(out_name,"r")
        out  = out_file.read()
        if flex is False:
            Ok   = "Ok"
        else:
            Ok = self.OREGX['CnvOk'].search(out)
        Stop = self.OREGX['failed'].search(out)
        Err  = self.OREGX['Error'].search(out)
        if (Err is not None) or ((Ok is None) and (Stop is not None)):
            # run was terminated by error or because of convergence, do not return anything
            energy = self.notconv
        else:
            if flex:
                try:
                    energy = float(self.OREGX['lastE'].search(out).group(3))
                except:
                    print("energy not found in ", out_name)
                    print(self.OREGX['lastE'].search(out))
                    #raise ValueError 
                    energy = self.notconv
            else:
                try:
                    energy = float(self.OREGX['singlepE'].search(out).group(1))
                except:
                    print("energy not found in ", out_name)
                    print(self.OREGX['singlepE'].search(out))
                    energy = self.notconv
                    #raise ValueError 
            if flex and energy != self.notconv:
                X = self.OREGX['fgeom'].search(out).group(1)
                data = list(map(lambda x: x.split(), X.splitlines()))
                cartesian = list()
                for d in data:
                    cartesian.append(d[:-1])
                cartesian = bohr2ang * np.asarray(cartesian, dtype='float')
                #check for clashes
                Dist = sp.spatial.distance.pdist(cartesian)
                if Dist.min() <= self.cutoff:
                    energy = self.notconv
                else:
                    if self.sptype == "clust":
                        structure.set_coordinates(cartesian)
                    else:
                        structure.update_internal_coords(cartesian)
        structure.set_energy(energy)
        return energy
    
    def fitcalc(self, structure, **kwargs):
        """
        Fitness calculator:
            - called on a cluster instance
            - creates xTB input files with modified zmat
            - calls xTB or just generates output structure and 
              returns a random number ("fake fitness")
            - parses log file and get energy and optimized geometry
            - filename created from ID
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
            raise ValueError("xTBParser: too many input files")
        xtb_namespace = self.filename + "_" + str(xnum) + str(self.ID)
        coord_name = xtb_namespace + ".coord"
        out_name = xtb_namespace + ".out"
        
        ### write values in input file
        flex = kwargs['flex']
        if flex:
            flex = '--opt'
        else:
            flex = ''
        if self.sptype == "clust":
            ZM = structure.dump_coords()
        else:
            ZM = structure.dump_coords(xtb=True)
        if self.run_esc:
            self.writeinp(ZM, coord_name)
        #set ID
        structure.set_ID(self.ID)

        # call esc or just return geom
        if self.run_esc:
            energy = self.call_esc(xtb_namespace, coord_name, out_name, structure, flex)
        else:
            energy = np.random.rand()
            if self.sptype == "clust":
                atoms = structure.atoms
                xyz = structure.get_coordinates()
            else:
                atoms = structure.zmat.Atoms
                xyz = structure.zmat.calc_coords()
            out_name = xtb_namespace + ".xyz"
            cluster_utils.write_xyz(atoms, xyz, out_name)

        #return here
        return energy, structure.get_coordinates()    
