# G Mancini Oct 2021

import numpy as np    
import os
import re
import scipy as sp
import subprocess as sub 

import cluster_utils
import sa_reacpair
import zmatrix

import gau_parser
import xtb_parser

# perform a xTB calculation then convert data to call G16

class double_caller:

    def __init__(self, **kwargs):
       """
       - g16caller: caller object for gaussian 16
       - xtbcaller: caller object for xTB
       """
        prop_defaults = {
            "ID"         : 0,
            "g16caller"  : False
            "xtbcaller"  : False
        }
        for (prop, default) in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))      
        if not self.g16caller: 
           raise ValueError("g16 caller object not provided")
        if not self.xtbcaller:
            raise ValueError("xTB caller object not provided")

    def call_esc(self, xxx):
        """
        call xTB, get energy and coordinates and call g16
        """
        xtbcaller.writeinp()

