"""
Container module for checks
"""

from mendeleev import element
import numpy as np
import scipy as sp


class ClashCheck():
    """
    Simple class to check the connectivity of a new geometry wrt of a reference
    structure
    This should be changed and use proxima
    """
    def __init__(self, atnums, refcrd, thrs=0.4, tol=0.):
        """
        atnums: list of atomic number
        refcrd: reference atomic crd in Angstrom as natom x 3 matrix
        thrs: threshold to compute the connectivity matrix
        tol: tolerance
        """
        #self._atn = [element(x).atomic_number for x in atnums]
        self._atn = list()
        for x in atnums:
            if len(x) == 1:
                el = x.upper()
            elif len(x) == 2:
                el = x[0].upper() + x[1]
            else:
                print("topocheck: read element ",x)
                raise ValueError("I don't know about this one")
            self._atn.append(element(el).atomic_number)
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
        return self._compute_conn(crd)
