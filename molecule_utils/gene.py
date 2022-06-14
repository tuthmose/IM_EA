import numpy as np

def cptospher(cpcrd):
    qval = np.sqrt(cpcrd[0][0]**2+cpcrd[0][1]**2)
    theta = np.arctan2(cpcrd[0][0]/qval, cpcrd[0][1]/qval)
    return (qval, theta, cpcrd[1][0])

def sphertocp(*spher):
    assert spher[1] < np.pi + 1e-6
    assert spher[1] > 0 - 1e-6
    return (np.array([spher[0]*np.sin(spher[1]), spher[0]*np.cos(spher[1])]), np.array([spher[2]]))

class Gene():

    def __init__(self, locus, allele=None):
        self._locus = locus
        self._allele = allele
        # self._allele = allele
        return None
    
    def mutate(self):
        return None

    def setallele(self, allele):
        self._allele = allele
    
class DihGene(Gene):

    def __init__(self, locus, allele=None):
        super().__init__(locus, allele)
        self._mutationpoint = 1

    def mutate(self, allele, mpoint=0):
        return allele
        # newgeom = self._chr.update_dihedrals(value, self._locus)
        # self._chr.updatecrd_fromxyz(newgeom)

class FiveRingGene(Gene):
    def __init__(self, locus, allele=None, fixq=True):
        """Five Member ring gene

        Args:
            locus (int): the ring index in the molecule
            fixq (bool, optional): Fix the Q degree of freedom of the ring. Defaults to True.
        """
        super().__init__(locus, allele)
        self._mutationpoint = 1 if fixq else 2

    def _mutateq(self, value):
        # Q ha senso fra 0 e 0.7 direi
        # suppongo periodicità 0-2pi
        toq = lambda x : x * 0.7 / np.pi
        return [[toq(value)], self._allele[1]]

    def _mutatetheta(self, value):
        # theta ha senso fra 0 e 2pi
        # Da controllare
        return [self._allele[0], [value]]

    def _getfunction(self, indx):
        if indx == 0:
            return self._mutatetheta
        if indx == 1:
            return self._mutateq

    def mutate(self, allele, mpoint=0):
        _function = self._getfunction(mpoint % self._mutationpoint)
        return _function(allele)


class SixRingGene(Gene):
    def __init__(self, locus, allele=None, fixq=True):
        """Six Member ring gene

        Args:
            locus (int): the ring index in the molecule
            fixq (bool, optional): Fix the Q degree of freedom of the ring. Defaults to True.
        """
        super().__init__(locus, allele)
        self._mutationpoint = 2 if fixq else 3

    def _mutateq(self, value):
        # Q ha senso fra 0 e 0.8?
        # suppongo periodicità 0-2pi
        toq = lambda x : x * 0.8 / np.pi
        return sphertocp(toq(value), self._allele[1], self._allele[2])

    def _mutatetheta(self, value):
        # theta ha senso fra 0 e pi
        totheta = lambda x: np.arccos(np.cos(x))
        return sphertocp(self._allele[0], totheta(value), self._allele[2])

    def _mutatephi(self, value):
        # Phi has 0 2pi periodicity
        return sphertocp(self._allele[0], self._allele[1], value)
     
    def _getfunction(self, indx):
        if indx == 0:
            return self._mutatetheta
        elif indx == 1:
            return self._mutatephi
        elif indx == 2:
            return self._mutateq

    def mutate(self, allele, mpoint=0):
        _function = self._getfunction(mpoint % self._mutationpoint)
        return _function(allele)

class Chromosome():

    def __init__(self, genes):
        self._genes = genes

class MolecularSystem(Chromosome):

	def __init__(self, genes, coordinates):
            self._genes = genes
            self._coordinates = coordinates

	def _update_coordinates(self):
            return None

	def change_allele(self, gene, allele):
            gene.mutate(allele)
            #self._chr.updatecrd_fromxyz(newgeom)
            self._update_coordinates()    

    