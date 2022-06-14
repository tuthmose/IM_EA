import numpy as np
import scipy as sp
import sys
sys.path.append("/home/gmancini/Dropbox/appunti/SearchAlgos/molecule_utils")
import zmatrix, ga_utils

print("Usage: python3 tpl2xyz file.tpl, filename.xyz")
myZM = zmatrix.zmat(sys.argv[1],rot_bonds=None,fmt="gau")
print("Rotatable bonds: ",myZM.rot_bonds)
print("Check clash:",myZM.check_clash(0.7))
myZM.write_file(sys.argv[2],fmt="xyz")
quit()
