# G Mancini Oct 21

import numpy as np
import scipy as sp

from skopt.sampler import Lhs
from skopt.space   import Space

# custom modules
import ga_evolution, ga_population, ga_utils
import cluster_utils
import zmatrix
import gau_parser
import xtb_parser
from topocheck import ClashCheck

def init_pop_dihedral(nchrm, INP, genes, cutoff, verbose, dihedral_space, \
    notconv, useXTB, useLH, caller):

    print("+++ CREATING POPULATION")
    chrm    = list()
    Zmatlist = list()

    # first element of pop is template
    ngenes = len(genes)
    zmat = ga_utils.linear_molecule(INP,genes,cutoff=cutoff)
    # Check against the template topology
    clashchecker = ClashCheck(zmat.zmat.Atoms, zmat.zmat.xyz)
    init_dih = zmat.get_chromosome()
    E, edih = caller.fitcalc(zmat,flex=False)
    chrm.append(edih)
    zmat.set_fitness(E)
    Zmatlist.append(zmat)
    print("+++ Template dihedrals: ",init_dih)

    # generate Latin Hypercube of dihedral values and sort wrt chebyshev distance
    redundancy_factor = 5
    space = Space([(-180., 180.) for i in range(len(genes))])
    lhs = Lhs(lhs_type="classic", criterion="ratio")
    x  = lhs.generate(space.dimensions, nchrm*redundancy_factor)
    xD = sp.spatial.distance.squareform(sp.spatial.distance.pdist(x,metric="chebyshev"))
    xS = np.sum(xD,axis=0)
    x  = np.asarray(x)
    x  = x[np.argsort(xS)][::-1]
    dih_iterator = iter(x)
    
    # test the generated values one by one and
    # select new values if needed
    e_count = 1
    z_count = 1
    N = list(range(ngenes))
    init_cutoff = cutoff

    if verbose >= 4:
        print("Initial structures")
        print("Energy, Dihedrals")

    # chromosome loop 
    for i in range(1, nchrm):
        E = notconv
        e_trial = 0
        # energy control loop
        while E == notconv and e_trial <= 30:
            e_trial += 1
            if useLH: 
                try:
                    mydihs = next(dih_iterator)
                except StopIteration:
                    mydihs = dihedral_space(ngenes)
            else:
                mydihs = dihedral_space(ngenes)
            Z = zmatrix.zmat(INP, genes, fmt="gau")
            out = Z.update_dihedrals(mydihs,N,update=True)
            schk = Z.check_clash(init_cutoff)
            tpck = clashchecker.check_geom(Z.xyz)
            clash = schk | tpck
            # clash control loop
            if clash:
                cl_trial = 0
                while clash and cl_trial <= 10:
                    if useLH: 
                        try:
                            mydihs = next(dih_iterator)
                        except StopIteration:
                            mydihs = dihedral_space(ngenes)
                    else:
                            theta = dihedral_space(ngenes)
                    Z = zmatrix.zmat(INP, genes, fmt="gau")
                    out = Z.update_dihedrals(theta,N,update=True)
                    schk = Z.check_clash(init_cutoff)
                    tpck = clashchecker.check_geom(Z.xyz)
                    clash = schk | tpck
                    if not clash:
                        mydihs = theta
                    cl_trial += 1
                z_count = z_count + cl_trial
                # end of clash control loop
            if clash:
                zmat = ga_utils.linear_molecule(INP, genes, cutoff=init_cutoff)
            else:
                zmat = ga_utils.linear_molecule(INP, genes, alleles=mydihs, cutoff=init_cutoff)
            E, edih = caller.fitcalc(zmat, flex=False)
            # end of energy control loop
        # add the template if no reasonable structure was generated
        if E == notconv:
            zmat = ga_utils.linear_molecule(INP, genes, cutoff=init_cutoff)
            E, edih = caller.fitcalc(zmat, flex=False)
        e_count = e_count + e_trial
        chrm.append(edih)
        zmat.set_fitness(E)
        Zmatlist.append(zmat)
        # end of chromosome loop

    # generate np arrays and convert to population
    chrm  = np.asarray(chrm)
    f = np.asarray(ga_utils.get_all_fitness(Zmatlist))
    mypop = ga_population.GA_population(chromosomes=chrm,fitness=f,\
    specimens=Zmatlist,template=INP, genes=genes)

    print("Generated ", z_count," to avoid clashes")
    print("Evaluated ", e_count," structures to create initial population")
    if verbose > 4:
        print("generated specimen ", mypop.chromosomes,mypop.fitness)
    np.savetxt("init_pop.dat",mypop.chromosomes)
        
    return mypop

def init_pop_cluster(nchrm, INP, genes, cutoff, verbose, notconv, Ctpl,\
    cmat, AT0, AT1, NAT, shapechanger, cross_over, caller, caller_type,\
    useXTB):
    
    print("+++ CREATING POPULATION")
    # create population; chromosomes are created randomly (checking 
    # that there are no collisions and single point converges)

    Xtpl = caller.initX
    nobjs = (len(Ctpl.atoms) - AT0)//NAT

    # generate Latin Hypercube of angles for rotation
    aspace = Space([(-180., 180.) for i in range(nobjs)])
    lhs = Lhs(lhs_type="classic", criterion="maximin")
    xa = lhs.generate(aspace.dimensions, nchrm)

    # generate Latin Hypercube of displacements
    dspace = Space([(0., 10.) for i in range(nobjs)])
    lhs = Lhs(lhs_type="classic", criterion="maximin")
    xd = lhs.generate(dspace.dimensions, nchrm)

    # generate Latin Hypercube of angles for orbit
    aspace2 = Space([(-10., 10.) for i in range(nobjs)])
    lhs = Lhs(lhs_type="classic", criterion="maximin")
    xa2 = lhs.generate(aspace.dimensions, nchrm)

    chrm    = list()
    Zmatlist = list()
    count = 1

    for i in range(nchrm):
        trial = 1
        E = notconv
        while E == notconv:
            trial += 1
            if trial > 10:
                raise ValueError("too many trials")
            # rattle
            X1 = cluster_utils.rattle(Xtpl, AT1, NAT, 0.1, 0.05, 20,
                                      check=cmat.check_geom)
            # rotate; sample from LH
            X2 = cluster_utils.rotate_coords(X1, AT1, NAT, xa[i], 0., 20,
                                             Ctpl.atmass, check=cmat.check_geom)
            # displace; sample from LH
            X3 = cluster_utils.displace(X2, AT1, NAT, xd[i], 0.0, 20,
                                        check=cmat.check_geom)
            # orbit; sample from LH
            X4 = cluster_utils.orbit(X3, AT1, NAT, xa2[i], 0.0, 20, Ctpl.atmass,
                                      check=cmat.check_geom)
            if useXTB:
                zmat = cluster_utils.xtb_cluster(X4, INP, genes)
            else:
                zmat = cluster_utils.gau_cluster(X4, INP, genes)
            E, X = caller.fitcalc(zmat, flex=False)
            count = count + 1
        chrm.append(X)
        zmat.set_fitness(E)
        Zmatlist.append(zmat)
        
    chrm  = np.asarray(chrm)
    f = np.asarray(ga_utils.get_all_fitness(Zmatlist))
        
    population = ga_population.GA_population(chromosomes=chrm, fitness=f, specimens=Zmatlist, \
    template=INP, genes=genes, sptype='cluster', caller_type=caller_type)

    print("Generated ", count," structures to create initial population")
    if verbose >= 3:
        print("generated specimen ", population.chromosomes, population.fitness)
        
    return population
