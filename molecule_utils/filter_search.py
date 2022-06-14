import math
import numpy as np
import scipy as sp
from collections import deque
from scipy import constants

kB = 0.001*sp.constants.physical_constants['Boltzmann constant'][0]\
*constants.physical_constants['Avogadro constant'][0]

har2kjmol = sp.constants.physical_constants["Avogadro constant"][0]\
*sp.constants.physical_constants["Hartree energy"][0]/1000.

class filter_search:
    """
    From an EA search pick "unique" and relevant structures. Filter is based on:
        1. transform energy as exp(-beta*E); remove all structures above probability
        threshold
        2. build RMSD graph and remove all edges with rmsd > cutoff;  in each 
        region with more than one node select centroid
        3. all singles and centroids within energy and RMSD are now included
        4. check rotational constants
    """
    
    def __init__(self,**kwargs):
        """
        rmsd and energy are mandatory arguments,
        clusters and label are optional (if no
        1st level separation was done)
        nselect is at least 2 (centroid and 
        energy minimum)
        """
        prop_defaults = {
            "rmsd"      : None,
            "energy"    : None,
            "label"     : None,
        }
        for (prop, default) in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))        
        assert isinstance(self.rmsd,np.ndarray)
        assert isinstance(self.energy,np.ndarray)
        assert self.rmsd.shape[0] == self.rmsd.shape[1]
        assert self.energy.shape[0] == self.rmsd.shape[1]
        assert isinstance(self.label,int) or isinstance(self.label,np.int64)
        self.nframe = self.rmsd.shape[1]    
