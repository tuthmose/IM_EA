from ga_population import *
from math import ceil

import numpy as np
import scipy as sp

def LinearSearch(pop, F_last, B_last, I_last, ffunc, fitkwd, fhigh):
    """
    Do a linear search across the whole population to improve
    current best solution
    The genotype of the best specimen is changed trying all other 
    genes from other specimens one at time
    """
    S = pop.get_size()
    G = pop.ngene
    best  = pop.chromosomes[0]
    fbest = pop.fitness[0]
    Tmp_pop = GA_population(template=pop.template,\
        coding_only=pop.coding_only,\
        genes=pop.genes)
    count = 0
    for j in range(1,S):
        for k in range(G):
            tmp = np.copy(best)
            if tmp[k] != pop.chromosomes[j][k]:
                tmp[k] = pop.chromosomes[j][k]
                Tmp_pop.spawn(1,tmp)
                count += 1
    if Tmp_pop.get_size() == 1:
        Tmp_pop.chromosomes = np.expand_dims(Tmp_pop.chromosomes,0)
    if Tmp_pop.get_size() > 0:
        Tmp_pop.calc_fitness(ffunc,Tmp_pop.get_size(),fitkwd)
        Tmp_pop.sortpop(fhigh)
        if (Tmp_pop.fitness[0] < fbest and  fhigh==False) or \
            (Tmp_pop.fitness[0] > fbest and fhigh==True):
            pop.merge(Tmp_pop)
            pop.sortpop(fhigh)
            pop.kill(count)
    del Tmp_pop
    bestfit, bestchrm, bestID = pop.get_best()
    return bestfit, bestchrm, bestID

    
def CataclysmicMutation(pop, F_last, B_last, I_last, ffunc, fitkwds, fhigh, nloci, genotype):
    """
    Create a new population from the best nbest chromosomes
    and do a simulated annealing-like random search
    See Brain and Addicoat, 10.1063/1.3656323
    """
    pstart = 0.05
    pincr  = 0.05
    pmax   = 0.4
    nbest  = 1
    cstart = pop.chromosomes[:nbest]
    fstart = pop.fitness[:nbest]
    #sstart = self.Isles[0].specimen[:nbest]
    newpop = GA_population(template=pop.template,coding_only=pop.coding_only,\
        genes=pop.genes)
    S = 0
    while S < pop.get_size():
        newpop.spawn(1,cstart,newfit=fstart)
        S = newpop.get_size()
    pmut  = pstart
    ncrhm = newpop.get_size()
    while pmut <= pmax:
        for chrm in range(ncrhm):
            for gene in range(newpop.ngene):
                coin = np.random.rand()
                if coin <= pmut:
                    newallele = genotype[np.random.choice(nloci)]
                    newpop.chromosomes[chrm][gene] = newallele                    
        newpop.calc_fitness(ffunc,newpop.get_size(),fitkwds)
        newpop.sortpop(fhigh)
        if newpop.fitness[0] < F_last and fhigh == False:
            break
        elif newpop.fitness[0] > F_last and fhigh == True:
            break
        pmut += pincr
    newpop.sortpop(fhigh)
    bestfit, bestchrm, bestID = newpop.get_best()
    return bestfit, bestchrm, bestID, pmut
