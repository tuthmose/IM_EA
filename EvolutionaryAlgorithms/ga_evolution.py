from copy import deepcopy as deepc
from ga_utils import *
from ga_population import *
from ga_restart import *
from cluster_utils import *
from math import ceil

import numpy as np
import scipy as sp
import sklearn.preprocessing as preprocessing
import zmatrix as zm

# Giordano Mancini Dec 2018
# As of January 19:

# island model working, using best
# individual migration

# removed self stop and evaluation of average fitness
# keeping CataclysmicMutation right now even with
# unsatisfactory result
# but it must be called separately, at variance with LS

min_max_scaler = preprocessing.MinMaxScaler()

class GenAlg:
    
    def __init__(self,**kwargs):
        """
        instatiate GA; keywords contain defaults
        sel_press  fraction of specimens to replace at each generation 
        sel_meth   parent selection method (rank, roulette, tournament)
        tsize      tournament size
        last_rank  use rank selection in last iterations        
        co_meth    crossover method (uniform, heuristic, 1point, 2 point, SBX)
        pCo        crossover probability
        alpha      alpha for heuristic crossover
        eta_sbx    eta for SBX crossover
        ppmut      probability of mutation for parents
        pcmut      probability of mutation for children
        mut        mutation method (constant, uniform, cluster)
        verbose    verbosity level
        seed       random seed
        cofunc     custom cross over function (needs co_meth=custom)
        mfunc      custom mutation function (needs mut=custom)
        savechrm   if True all eliminated chromosomes are saved in a history array
                   which at the end is merged with the current population
        hof        Hall of Fame the individuals contained in the hall of fame object are directly injected into the next generation without Operators acting

        """
        # check defaults
        prop_defaults = {
            "sel_press" : 0.33,
            "sel_meth"  : 'tournament',
            "tsize"     : 2,
            "last_rank" : False,
            "co_meth"   : 'uniform',
            "pCo"       : 0.5,
            "alpha"     : 0.5,
            "eta_sbx"   : 3.0,
            "ppmut"     : 0.1,
            "pcmut"     : 0.2,
            "mut"       : "constant",
            "verbose"   : 1, 
            "seed"      : None,
            "mfunc"     : None,
            "cofunc"    : None,
            "savechrm"  : False,
            "nisle"     : 1,
            "migr_freq" : 0,
            "cutoff"    : 0.85,
            "debug"     : False,
            "hof"       : 0.00,
            "kpco"      : 0.2
        }
        for (prop, default) in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))      
        assert self.pcmut <= 0.99
        assert self.ppmut <= 0.99
        assert self.sel_meth in ("tournament","rank","roulette")
        assert self.co_meth in ("uniform","heuristic","SBX","custom","rotation",\
            "ordered","ordered2","1p","2p","kp")
        assert self.mut in ("uniform","constant","custom","rotation","swap",\
            "shuffle","rotation2","rotation3")
        if self.seed is not None:
            assert isinstance(1.0*self.seed,float)
            np.random.seed(self.seed)
        if self.verbose >= 2:
            print("+++ Parent selection method: ",self.sel_meth)
            print("+++ CrossOver method: ",self.co_meth)
            print("+++ Mutation  method: ",self.mut)
            
    def Split(self):
        """ 
        Split population to apply Island models
        """
        Isles = list()
        sub_size = int(self.pop.get_size() / self.nisle)
        for i in range(self.nisle):
            #create an empty population in each isle
            Isles.append(GA_population(template=self.pop.template,\
                coding_only=self.pop.coding_only,ngene=self.pop.ngene,\
                genes=self.pop.genes,sptype=self.pop.sptype,\
                caller_type=self.pop.caller_type))
            ## add specimens
            Isles[i].spawn(sub_size,self.pop.chromosomes[i*sub_size: (i+1)*sub_size])
        del self.pop
        return Isles
    
    def Join(self):
        """
        Form a Unique population from a list of Isles
        """
        if self.nisle == 0:
            raise ValueError("Cannot join without Isles")
        elif self.nisle == 1:
            return self.Isles[0]
        C = list()
        F = list()
        S = list()
        for i in range(self.nisle):
            C.append(self.Isles[i].chromosomes)
            F.append(self.Isles[i].fitness)
            if self.Isles[0].coding_only is False:
                S = S + self.Isles[i].specimens
        C = np.asarray(C)
        F = np.asarray(F)
        C.shape = tuple([sum([i.get_size() for i in self.Isles])] + \
            list(self.Isles[0].chromosomes[0].shape))
        F = np.ravel(F)
        pop =  GA_population(template=self.Isles[0].template,coding_only=self.Isles[0].coding_only,\
                genes=self.Isles[0].genes,chromosomes=C,fitness=F,specimens=S)
        pop.sortpop(self.fhigh)
        return pop
    
    def Get_Best_Isles(self, best_highest):
        """
        From a list of Isles (subpopulations)
        get best specimens
        """
        pop_rank = list()
        for i in range(self.nisle):
            isle_best = self.Isles[i].get_best()
            pop_rank.append(isle_best[0])
        order = np.argsort(pop_rank)
        if best_highest is True:
            order = order[::-1]
        tmp = [self.Isles[x] for x in order]
        self.Isles = tmp
        return self.Isles[0].get_best()
    
    def Migration(self):
        if self.mpolicy == "round":
            self.RoundRobin()
        elif self.mpolicy == "maxdist":
            if self.distance is None:
                raise ValueError("Migration: provide a metric for distance evaluation")
            self.MaxDist()
        else:
            raise ValueError("Unknown migration policy")
        for i in range(self.nisle):
            self.Isles[i].sortpop(self.fhigh)
    
    def RoundRobin(self):
        """
        Migrate best two chromosomes between islands
        in a round robin fashion
        """
        for i in range(1,self.nisle):
            self.Isles[i].kill(2)
            if self.Isles[i].sptype == 'cluster':
                self.Isles[i].spawn(1,self.Isles[i-1].specimens[0].X,self.Isles[i-1].fitness[0])
                self.Isles[i].spawn(1,self.Isles[i-1].specimens[1].X,self.Isles[i-1].fitness[1])
            else:
                self.Isles[i].spawn(1,self.Isles[i-1].chromosomes[0],self.Isles[i-1].fitness[0])
                self.Isles[i].spawn(1,self.Isles[i-1].chromosomes[1],self.Isles[i-1].fitness[1])
        self.Isles[0].kill(2)
        if self.Isles[0].sptype == 'cluster':
            self.Isles[0].spawn(1,self.Isles[-1].specimens[0].X,self.Isles[-1].fitness[0])
            self.Isles[0].spawn(1,self.Isles[-1].specimens[1].X,self.Isles[-1].fitness[1])
        else:
            self.Isles[0].spawn(1,self.Isles[-1].chromosomes[0],self.Isles[-1].fitness[0])
            self.Isles[0].spawn(1,self.Isles[-1].chromosomes[1],self.Isles[-1].fitness[1])
        for i in range(self.nisle):
            self.Isles[i].sortpop(self.fhigh)
                    
    def MaxDist(self):
        best = np.empty((self.nisle,self.Isles[0].ngene))
        for i in range(self.nisle):
            best[i] = self.Isles[i].get_best()[1]
        D = sp.spatial.distance.pdist(best,metric=self.distance)
        D = sp.spatial.distance.squareform(D)
        for i in range(self.nisle):
            m = np.argmax(D[i])
            self.Isles[i].kill(1)
            self.Isles[i].spawn(1,self.Isles[m].chromosomes[0],self.Isles[m].fitness[0])
            
    def Start(self,**kwargs):
        #TODO: put all mandatory here and all numeric with defaults
        # in __init__
        """
        Complete GA setup with another set of keywords / defaults.
        Note that many defaults are None but are actually mandatory.
        genotype  : iterable with available values for creating genes (None)
        pop       : population  (None)
        niter     : number of generations  (0)
        ffunc     : fitness function object (None)
        fitkwds   : keywords for the fitness function (None)
        fhigh     : if False better means lower or negative (False)
        optimal   : best possible fitness (for benchmarking)  (None)
        tol       : tolerance to consider the GA converged to optimal (1e-9)
        nisle     : number of islands; default: no islands (1)
        migr_freq : migration frequency between islands in generations (0);
                    default: do not migrate
        LS        : do a Linear Search of chromosomes at the end (False)
        replace   : replace best chromosome with previous best one if better (False)
        distance  : distance metric for maxdist migration policy (euclidean)
        mpolicy   : migration policy (round robin)
        split_gen : split genotype with Island model (False)
        """
        # check defaults
        prop_defaults = {
            "genotype"  : None,
            "pop"       : None,
            "niter"     : 0,
            "ffunc"     : None,
            "fitkwds"   : None,
            "fhigh"     : False,
            "optimal"   : None,
            "tol"       : 1e-9,
            "LS"        : False,
            "replace"   : False,
            "distance"  : "euclidean",
            "mpolicy"   : "round",
            "split_gen" : False,
            "force_ls"  : False,
            "nisle"     : 1
            }
        for (prop, default) in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))     
        ## setup
        if isinstance(self.genotype,list) or isinstance(self.genotype,tuple):
            self.genotype = np.asarray(self.genotype)
        if isinstance(self.genotype,np.ndarray):
            self.nloci = self.genotype.shape[0]
        else:
            self.nloci = None
        assert self.niter >= 1
        assert self.pop.get_size() % 2 == 0
        self.init_size = self.pop.get_size()
        
        # caller_type
        self.caller_type = self.pop.caller_type
        #Island model
        if self.nisle > 1:
            #print("Pop size={0:d}, nisle={1:d}".format(self.pop.get_size(), self.nisle))
            assert self.pop.get_size() / self.nisle > 2
        if self.split_gen == True:
            if not np.any(self.genotype):
                raise ValueError('Island model: cannot split with no genotype')
        if self.nisle > 1:
            genotypes = list()
            if self.pop.get_size() % self.nisle != 0:
                print("Pop size={0:d}, nisle={1:d}".format(self.pop.get_size(), self.nisle))
                raise ValueError("pop_size % nisle !=0")
            if self.verbose:
                print("Splitting population in {0:4d} isles".format(self.nisle))
            self.Isles = self.Split()
            if not np.any(self.genotype):
                genotypes = [None]
            elif self.split_gen == False:
                genotypes = [self.genotype]
            else:
                genotypes = list()
                ngen = self.Isles[0].chromosomes[0].shape[0] // self.nisle
            for i in range(self.nisle):
                self.Isles[i].calc_fitness(self.ffunc,self.Isles[i].get_size(),self.fitkwds)
                self.Isles[i].sortpop(self.fhigh)
                if self.split_gen:
                    genotypes.append(self.genotype[i*ngen:(i+1)*ngen])
            self.previous = (self.Get_Best_Isles(self.fhigh))
            if self.split_gen and self.genotype is not None and self.nloci % self.nisle != 0:
                genotypes[0] = np.append(genotypes[0],self.genotype[self.nisle*ngen:])
        elif self.nisle == 1:
            self.Isles = [ self.pop ]            
            self.Isles[0].calc_fitness(self.ffunc,self.Isles[0].get_size(),self.fitkwds)
            self.Isles[0].sortpop(self.fhigh)
            self.previous = (self.Isles[0].get_best())
            del self.pop
            genotypes = [self.genotype]
        else:
            raise ValueError
            
        #create empty pop if all chromosomes must be saved
        if self.savechrm == True:
            self.history = GA_population(coding_only=True)

        if self.verbose:
            if self.nisle <= 1:
                print("+++ Starting selection: %d specimens replaced at each generation" %\
                    (ceil(self.sel_press*self.Isles[0].get_size())))
               #if self.verbose > 2:
               #    print("+++ Initial conditions:", self.previous[0], self.previous[1],\
               #        self.previous[2])
               #else:
               #    print("+++ Initial conditions:", self.previous[0])
            else:
                print("+++ Starting selection: %d specimens replaced at each generation in each Isle" %\
                    (ceil(self.sel_press*self.Isles[0].get_size())))
                self.previous = (self.Get_Best_Isles(self.fhigh))
                if self.verbose > 2:
                    print("+++ Initial conditions:", self.previous[0], self.previous[1],\
                        self.previous[2])
                else:
                    print("+++ Initial conditions:", self.previous[0])
        return genotypes
    
    def Evolve(self,**kwargs):
        """
        call Start to set mandatory arguments and start selection
        return lists of best fitness and chromosomes (+ specimen)
        at each generation
        """
        genotypes = self.Start(**kwargs)
        
        F  = list()
        B  = list()
        I  = list()
        
        solution_found = False
            
        #--------- main loop
        for it in range(self.niter):
            self.it = it
            if self.last_rank and (self.niter-it)<0.1*self.niter:
                self.sel_meth = "rank"
            bestfit, bestchrm, bestID = self.NextGeneration(it, genotypes)
            if self.verbose >= 4:
                print("+++ Generation {0:d} done".format(it))
                               
            #update summary
            F.append(bestfit)  
            B.append(bestchrm)
            I.append(bestID)
            if self.verbose > 1 and it%10==20:
                print("\tBest fit %f at generation %d " % (bestfit,it))            
            #check termination condition
            if self.optimal is not None:
                if (self.fhigh and bestfit >= self.optimal - self.tol):
                    solution_found = True
                    print("+++ optimal solution found ; stopping after %d iterations " % it)  
                    break
                elif bestfit <= self.optimal - self.tol and self.fhigh==False:
                    solution_found = True
                    print("+++ optimal solution found ; stopping after %d iterations " % it)  
                    break
        #------- end main loop
        
        #if using Island model recreate unique population
        if self.nisle > 1:
            self.pop = self.Join()
        else:
            self.pop = self.Isles[0]
        self.pop.sortpop(self.fhigh)
        #if using history, add current pop to history
        if self.savechrm:
            self.history.merge(self.pop)
        
        #------- restart with LinearSearch
        if solution_found==False and self.verbose:
                print("+++ Maximum number of generations reached")
                
        if it >= 100:
            Fgrad = np.gradient(np.asarray(F[:ceil(-0.05*it)]))
            Fgrad = np.mean(Fgrad)
            if abs(Fgrad) < self.tol:
                restart = True
            else:
                restart = False
        else:
            restart = False
            
        if (solution_found==False and self.LS and restart) or (self.LS and self.force_ls):
            print("+++ Starting Linear Search from best fitness: {0:10.7f}".format(F[-1]))
            FLS, BLS, ILS = LinearSearch(self.pop, F, B, I, self.ffunc, self.fitkwds, self.fhigh)
            if (FLS > F[-1] and self.fhigh==False) or (FLS < F[-1] and self.fhigh==True):
                if self.verbose:
                    print("+++ Linear Search failed to improve solution")
            else:
                if self.verbose:
                    print("+++ Linear Search solution: {0:15.9f}".format(FLS))
                F[-1] = FLS
                B[-1] = BLS
                I[-1] = ILS
        elif solution_found==False and self.LS and restart==False:
            if self.verbose:
                print("+++ No need to start Linear Search: {0:10.7f}".format(Fgrad))
        return F, B, I
       
    def NextGeneration(self,it,genotypes):
        """ 
        Apply Genetic Operators in the following order:
            - ParentSelection (number of offsping to generate determined by selection pressure)
            - CrossOver
            - Mutation
            - Calculate fitness of children and mutated parents
            - Survivor selection
        """
        for i in range(self.nisle):
        # create new generation
            nmating, parents = self.ParentSelection(self.Isles[i])
            if self.debug:
                print("nmating",nmating,parents)
            children = GA_population(template=self.Isles[i].template,\
                coding_only=self.Isles[i].coding_only, genes=self.Isles[i].genes,\
                sptype = self.Isles[i].sptype, caller_type=self.caller_type)
            if self.hof != 0.:
                hall_of_fame = GA_population(template=self.Isles[i].template,\
                    coding_only=self.Isles[i].coding_only, genes=self.Isles[i].genes,\
                    sptype = self.Isles[i].sptype, caller_type=self.caller_type)
                nhof = max(ceil(self.hof*self.Isles[i].get_size()),2)
                if self.debug: print("hall of fame size ",self.hof, nhof)
                if self.Isles[i].coding_only == False:
                    for ihof in range(nhof):
                        hall_of_fame.insert_specimen(self.Isles[i].chromosomes[ihof],\
                        self.Isles[i].specimens[ihof],newfit=self.Isles[i].fitness[ihof])
                else:
                    for ihof in range(nhof):
                        hall_of_fame.insert_specimen(self.Isles[i].chromosomes[ihof],\
                        None,newfit=self.Isles[i].fitness[ihof])
                if self.debug: print("hal of fame fitness ",self.Isles[i].fitness[ihof])
            else:
                hall_of_fame = None
            self.CrossOver(nmating, parents, children, self.Isles[i])
            if self.debug:
                print("crossover",children.get_size(),children)
        #mutation is on parents AND children
            mutated  = self.Mutate(self.Isles[i],self.ppmut,it,genotypes,i,False)
            if self.debug:
                print("mutated parents",mutated)
            self.Mutate(children,self.pcmut,it,genotypes,i,True)
            #raise ValueError
        #Update fitness
            children.calc_fitness(self.ffunc,children.get_size(),self.fitkwds)
            self.Isles[i].calc_fitness(self.ffunc,mutated,self.fitkwds)
            self.Isles[i].sortpop(self.fhigh)
            children.sortpop(self.fhigh)
            #print(children.fitness,self.Isles[i].fitness,"fff",self.fhigh)
        # Selection
            bestfit, bestchrm, bestID = self.FitSelection(nmating, children, it, self.Isles[i], hall_of_fame)
            del children, hall_of_fame
        #end next gen within isles
        if self.nisle > 1:
            P = 0
            for p in self.Isles:
                P += p.get_size()
            if P != self.init_size:
                print(self.sel_press, self.pCo, self.ppmut, self.pcmut,self.init_size,P,it)
                raise ValueError
            
        if self.nisle > 1:
            if self.migr_freq > 0 and it % self.migr_freq==0:
                self.Migration()
            bestfit, bestchrm, bestID = self.Get_Best_Isles(self.fhigh)
        return bestfit, bestchrm, bestID
   
    def TournamentSelection(self, nmating, M, pop):
        parents = list()
        for i in range(nmating): 
        # selected random parents and sort self.tsize with best fitness
            competitors = np.random.choice(M, size=self.tsize)
            #print("competitors",competitors)
            added = False
            for i in competitors:
                if i in parents:
                    continue
                else:
                    parents.append(i)
                    added = True
                    break
            if added is False:    
                parents.append(competitors[0])
        return parents

    def RankSelection(self, nmating, pop):
        return list(range(nmating))
    
    def Roulette(self, allfit, pop):
        coin = np.random.uniform(high=allfit)
        fsum = 0.
        for i in range(pop.get_size()):
            fsum = fsum + pop.fitness[i]
            if fsum >= coin:
                return i
        assert False
    
    def RouletteWheelSelection(self, nmating, pop):
        parents = list()
        allfit = np.sum(pop.fitness)   
        for i in range(0, nmating-1, 2):
            parentA = 0
            parentB = 0
            while parentA == parentB:
                parentA = self.Roulette(allfit)
                parentB = self.Roulette(allfit)
            parents.append(parentA)
            parents.append(parentB)
        return parents

    def ParentSelection(self, pop):
        """
        Return a list of parents 
        Tournament: select 2 parents with K=pop_size//k
        Rank: select best nmating chromosomes
        Roulette:
        """
        nmating = ceil(self.sel_press * pop.get_size())
        if nmating % 2 !=0:
            nmating = nmating - 1
        M = tuple(pop.get_ch_labels())
        #if was 1 now is zero
        if nmating <= 1: 
            nmating = 2
            parents = list(range(len(pop.chromosomes)))        
        elif self.sel_meth == "tournament":
            parents = self.TournamentSelection(nmating,  M, pop)
        elif self.sel_meth == "rank":
            parents = self.RankSelection(nmating, pop)
        elif "roulette":
            parents = self.RouletteWheelSelection(nmating, pop)
        return nmating, parents    
    
    def OnePoint(self, p0, p1):
        """
        One point crossover
        """
        pbreak = np.random.choice(len(p0)-1)
        child0 = p0
        child0[pbreak:] = p1[pbreak:]
        child1 = p1
        child1[pbreak:] = p0[pbreak:]        
        return child0, child1
    
    def TwoPoint(self, p0, p1):
        """
        Two point crossover
        """
        pbreak = np.random.choice(len(p0)-1)
        lbreak = np.random.choice(len(p0)-pbreak)        
        ebreak = pbreak + lbreak
        child0 = p0
        child0[pbreak:ebreak] = p1[pbreak:ebreak]
        child1 = p1
        child1[pbreak:ebreak] = p0[pbreak:ebreak]        
        return child0, child1
    
    def KPoint(self, p0, p1, k=0):
        """
        K Point crossover with kmax=len(p0//2))
        K=0 falls to OnePoint
        """
        maxk = len(p0)//2
        child0, child1 = self.OnePoint(p0, p1)
        if k==0:
            return child0, child1
        elif k<=maxk:
            for ik in range(k):
                child0, child1 = self.OnePoint(child0,child1)
        else:
            raise ValueError("max K too high")
        return child0, child1

    
    def Uniform(self, pop, p0, p1, parents):
        """
        Uniform crossover:
        - generate two offspring with scattered crossover
        - use third parent for equal alleles.
        """
        k = int(np.random.choice(parents,1))
        p2 = pop.get_one(k)
        mask = np.random.randint(2,size=pop.ngene)
        mask[p0==p1] = 2
        # children
        child0 = np.where(mask==0,p0,p1)
        child1 = np.where(mask==0,p1,p0)
        child0[mask==2] = p2[mask==2]
        child1[mask==2] = p2[mask==2]
        return child0, child1
    
    def Heuristic(self,p0,p1):
        """
        Heuristic crossover for real valued genes:
        generate a+(1-a) w. average of non encoded genes        
        """   
        child0 = self.alpha*p0 + (1.-self.alpha)*p1
        child1 = (1.-self.alpha)*p0 + self.alpha*p1        
        return child0, child1
    
    def SBX(self, p0, p1):
        """
        apply simulated binary crossover (SBX)
        with parameter self.eta_sbx
        """
        mu   = np.random.rand()
        if mu <= 0.5:
            beta = 2.* mu **(1./(self.eta_sbx+1.))
        else:
            beta = (0.5*(1.- mu)) **(1./(self.eta_sbx+1.))
        child0 = 0.5* ((1+beta)*p0 + (1.-beta)*p1)
        child1 = 0.5* ((1-beta)*p0 + (1.+beta)*p1)
        return child0, child1
    
    def Rotation(self, pop, p0, p1):
        """
        try to mix dihedral angles directly on a
        molecule object checking for clashes
        p0 and p1 are labels
        """        
        #politically correct
        #create new molecule for child
        child = deepc(p0)
        maxtry = 20
        d0  = p0.get_chromosome() 
        d1  = p1.get_chromosome()
        out = p0.get_chromosome() 
        for i in range(pop.ngene):
            if d0[i] != d1[i]:
                theta = zm.deg_circ_mean([d0[i],d1[i]])
                delta = zm.deg_circ_dist(d0[i],theta)/maxtry
                clash = child.zmat.update_clash([theta],[i],self.cutoff)
                ntry = 0
                while clash and ntry <= maxtry:
                    ntry += 1
                    theta += delta
                    clash = child.zmat.update_clash([theta],[i],self.cutoff)
                if clash == False:
                    out[i] = theta
            else:
                pass
        xx = child.zmat.update_dihedrals(out,list(range(pop.ngene)),update=True)
        clash = child.zmat.check_clash(self.cutoff)
        D = child.zmat.mindist(True)
        if clash:
            #D =child.zmat.mindist(True)
            #raise ValueError("doing co",D,clash,out,child.get_chromosome())
            return p0
        else:
            return child
        
    def Ordered2(self, p0, p1):
        pbreak = np.random.choice(len(p0)-1)
        lbreak = np.random.choice(len(p0)-pbreak)
        child0      = -1*np.ones(len(p0), dtype='int')
        child1      = -1*np.ones(len(p1), dtype='int')
        for locus, al in enumerate(p0):
            if locus < pbreak or locus >= pbreak+lbreak:
                l1 = np.where(p1==al)[0][0]
                child0[locus] = p1[l1]
                l1 = np.where(p0==p1[locus])[0][0]
                child1[locus] = p0[l1]
            else:
                child0[locus] = p0[locus]
                child1[locus] = p1[locus]
        return child0, child1
        
    def Ordered(self, p0, p1):
        """
        Cross over method for permutation lists
        """
        maxlen = ceil
        if len(np.where(p0==p1)) == len(p0):
            #print("same ",p0,p1)
            return p0, p1
        else:
            pbreak = np.random.choice(len(p0)-3)
            lbreak = np.random.choice(len(p0)-pbreak-3)
            ebreak = pbreak + lbreak
            child0 = -np.ones(len(p0), dtype='int')
            child0[pbreak:ebreak] = p0[pbreak:ebreak]
            miss0 = iter(set(p1).difference(child0))
            child1 = -np.ones(len(p1), dtype='int')
            child1[pbreak:ebreak] = p1[pbreak:ebreak]
            miss1 = iter(set(p0).difference(child1))
            for locus in range(pbreak):
                child0[locus] = next(miss0) 
                child1[locus] = next(miss1)
            for locus in range(ebreak, len(p0)):    
                child0[locus] = next(miss0) 
                child1[locus] = next(miss1)
            if np.any(child0==-1) or np.any(child1==-1):
                print(child0,child1)
                raise ValueError
            if len(set(child0))!=len(p0) or len(set(child1))!=len(p1):
                print(child0,child1)
                raise ValueError
            # this may happenunder several, if unlikely, cases where the two parents
            # are similar and the substring is quite long thus leaving small room
            # for permutations
            #if len(np.where(p0==child0)[0])==len(p0):
            #    print(p0,p1,child0,pbreak,ebreak)
            #    raise ValueError
            #if len(np.where(p1==child1)[0])==len(p1):        
            #    print(p0,p1,child1,pbreak,ebreak)
            #    raise ValueError
        return child0, child1
            
    def CrossOver(self, nmating, parents, children, pop):
        """
        Determine cross-over events and call appropriated method
        """
        for i in range(0,nmating-1,2): 
            if self.co_meth == "rotation":
                p0 = pop.get_one_sp(parents[i])
                p1 = pop.get_one_sp(parents[i+1])
            else:
                p0 = pop.get_one(parents[i])
                p1 = pop.get_one(parents[i+1])
            coin = np.random.rand()
            if coin <= self.pCo:
                if self.co_meth == "uniform":
                    child0, child1 = self.Uniform(pop, p0, p1, parents)
                elif self.co_meth == "heuristic":
                    child0, child1 = self.Heuristic(p0, p1)
                elif self.co_meth == "1p":
                    child0, child1 = self.OnePoint(p0, p1)
                elif self.co_meth == "2p":
                    child0, child1 = self.TwoPoint(p0, p1)
                elif self.co_meth == "kp":
                    kp = int(self.kpco*len(p0))
                    child0, child1 = self.KPoint(p0, p1, kp)                    
                elif self.co_meth == "SBX":
                    child0, child1 = self.SBX(p0, p1)
                elif self.co_meth == "ordered":
                    child0, child1 = self.Ordered(p0, p1)
                elif self.co_meth == "ordered2":
                    child0, child1 = self.Ordered2(p0, p1)                    
                elif self.co_meth == "custom":
                    child0, child1 = self.cofunc(self.alpha, p0, p1)
                elif self.co_meth == "rotation":
                    child0 = self.Rotation(pop,p0,p1)
                    child1 = self.Rotation(pop,p1,p0)
            else:    
                child0 = p0
                child1 = p1
            #add to children population
            if self.co_meth == "rotation":
                children.insert_specimen(child0.get_chromosome(),child0)
                children.insert_specimen(child1.get_chromosome(),child1)
            else:
                children.spawn(2,[child0,child1])
                
    def Mutate_shuffle(self, population, prob):
        """
        Shuffle a sub set of chromosome values
        the number of mutations is prob*tot_genes
        """
        tot_genes = population.get_size()*len(population.chromosomes[0])
        num_scrambles = ceil(prob * tot_genes)
        nspec = num_scrambles//population.get_size()
        mut_spec = np.random.choice(population.get_size(), size=num_scrambles)
        for ms in mut_spec:
            pbreak = np.random.choice(len(population.chromosomes[ms])-1)
            lbreak = np.random.choice(len(population.chromosomes[ms])-pbreak)
            np.random.shuffle(population.chromosomes[ms][pbreak:pbreak+lbreak])
        return list(set(mut_spec))
                
    def Mutate_swap(self, population, prob):
        """
        swap an allele with its left or right neighbour;
        the number of mutations is prob*tot_genes
        """
        tot_genes = population.get_size()*len(population.chromosomes[0])
        num_swaps = ceil(prob * tot_genes)
        mut_spec = np.random.choice(population.get_size(), size=num_swaps)
        for ms in mut_spec:
            locus = np.random.choice(len(population.chromosomes[ms])-1)            
            sign = int(np.sign(np.random.rand()-0.5))
            tmp = population.chromosomes[ms][locus]
            population.chromosomes[ms][locus] = population.chromosomes[ms][locus+1*sign]
            population.chromosomes[ms][locus+1*sign] = tmp
        return list(set(mut_spec))

    def Mutate_rotation3(self, population, prob, newp):
        """
        Apply the mutation with a roulette inversely proportional
        to fitness (=random for children)
        The number of mutation is nspec*prob
        Check clash
        """
        num_mut   = ceil(prob * population.get_size())
        if not newp:
            transf = population.fitness.reshape(-1, 1)
            renorm_fit = min_max_scaler.fit_transform(transf)[:,0]
            total_fit  = np.sum(renorm_fit)
        else:
            renorm_fit= list(range(len(population.chromosomes)))
            total_fit = len(population.chromosomes)
        clist = population.get_ch_labels()#[::-1]
        #print(clist)
        nmut = 0
        while nmut < num_mut:
            coin = np.random.uniform(high=total_fit)
            #print(renorm_fit, population.fitness, coin, total_fit)
            fsum = 0.
            mutated = list()
            for chrm in clist:
                fsum = fsum + renorm_fit[chrm]
                #print(fsum, renorm_fit[chrm])
                if fsum >= coin and np.random.rand()<= prob:
                    gmut = np.random.choice(population.ngene)
                    maxtry = 20
                    step = 0
                    clash = True
                    test_spec = deepc(population.specimens[chrm])
                    while clash and step <= maxtry:
                        theta = self.genotype(1)
                        clash = test_spec.zmat.update_clash(theta,[gmut],self.cutoff)
                        step += 1
                    if clash is False:
                        population.specimens[chrm].set_chromosome(theta,[gmut])
                        mutated.append(chrm)
            #raise ValueError
            nmut += 1
        return mutated

    def Mutate_rotation2(self, population, prob):
        """
        """
        mutated = list()
        for chrm in population.get_ch_labels():
            coin = np.random.rand()
            if coin <= prob:
                maxtry = 30
                step = 0
                clash = True
                test_spec = deepc(population.specimens[chrm])
                mtype = np.random.rand()
                if mtype < 0.5:
                    while clash and step <= maxtry:
                        gmut = [np.random.choice(population.ngene)]
                        theta = self.genotype(1)
                        for i in population.chromosomes:
                            if (np.abs(theta[0] - population.chromosomes[chrm][gmut])<10.):
                                continue
                        clash = test_spec.zmat.update_clash(theta,gmut,self.cutoff)
                        step += 1
                    else:
                        gmut = np.random.choice(population.ngene,size=2)
                        theta = self.genotype(2)
                        for i in population.chromosomes:
                            if (np.abs(theta[0] - population.chromosomes[chrm][gmut[0]])<10.) or \
                               (np.abs(theta[1] - population.chromosomes[chrm][gmut[1]])<10.):
                                continue
                        clash = test_spec.zmat.update_clash(theta,gmut,self.cutoff)
                        step += 1
                    #   while clash and step <= maxtry:
                    #       sign = np.random.rand()-1
                    #       theta = [population.chromosomes[chrm][gmut] + sign*10.]
                    #       clash = test_spec.zmat.update_clash(theta,[gmut],self.cutoff)
                    #       step += 1
                if clash is False:
                    population.specimens[chrm].set_chromosome(theta,gmut)
                    mutated.append(chrm)
        return mutated
            
    def Mutate_rotation(self, population, prob):
        """
        mutate each specimen with probability prob
        and chose mutated gene uniformly using 
        zmat check clash
        """
        mutated = list()
        for chrm in population.get_ch_labels():
            coin = np.random.rand()
            if coin <= prob:
                gmut = np.random.choice(population.ngene)
                maxtry = 20
                step = 0
                clash = True
                test_spec = deepc(population.specimens[chrm])
                while clash and step <= maxtry:
                    theta = self.genotype(1)
                    clash = test_spec.zmat.update_clash(theta,[gmut],self.cutoff)
                    step += 1
                if clash is False:
                    population.specimens[chrm].set_chromosome(theta,[gmut])
                    mutated.append(chrm)
        return mutated
       
    def Mutate_const(self, population, prob, genotype):
        """
        mutate each specimen with probability prob
        and chose mutated gene uniformly
        """
        mutated = list()
        for chrm in population.get_ch_labels():
            coin = np.random.rand()
            if coin <= prob:
                mutated.append(chrm)
                if self.nloci is not None:
                    newallele = self.genotype[np.random.choice(self.genotype.shape[0])]
                else:
                    newallele = self.genotype(1)[0]
                population.mutate_specimen(chrm,newallele)
        return mutated       
             
    def Mutate_unif(self, population, prob):
        """
        Pick prob*ngene genes and mutate them with probability prob.
        On average, prob**2 will mutate
        """
        mutated = list()
        S = population.get_size()
        G = len(self.genotype)
        num_genes = S * G
        mut_genes = np.random.choice(num_genes, ceil(prob*num_genes))
        #print("mut unif", S,G,mut_genes.shape)
        for chrm in range(S):
            for gene in range(G):
                if chrm * gene in mut_genes:
                    coin = np.random.rand()
                    if coin <= prob:
                        mutated.append(chrm)
                        newallele = self.genotype[np.random.choice(self.nloci)]
                        population.chromosomes[chrm][gene] = newallele
        return mutated
    
    def Mutate(self, population, prob, it, genotypes, npop, newp):
        """
        Mutate according to selected method
        """
        if self.debug:
            print("Mutation function",self.mut)
        if self.last_rank and (self.niter-it)<0.05*self.niter:
            prob = 0.1 * prob
        if self.mut == "constant":
            if self.split_gen:
                G = genotypes[npop]
            else:
                G = genotypes[0]
            mutated = self.Mutate_const(population, prob, G)
        elif self.mut == "uniform":
            mutated = self.Mutate_unif(population, prob)
        elif self.mut == "rotation":
            mutated = self.Mutate_rotation(population, prob)
        elif self.mut == "rotation2":
            mutated = self.Mutate_rotation2(population, prob)
        elif self.mut == "rotation3":
            mutated = self.Mutate_rotation3(population, prob, newp)
        elif self.mut == "swap":
            mutated = self.Mutate_swap(population, prob)
        elif self.mut == "shuffle":
            mutated = self.Mutate_shuffle(population, prob)            
        elif self.mut == "custom":
            #print("custom mutation function",self.mfunc)
            mutated = self.mfunc(population, prob, newp)
        else:
            raise ValueError("Unknown mutation method")
        if self.debug:
            if mutated is not None:
                print("mutations",len(mutated),population.get_size(),prob)
        return mutated

    def FitSelection(self, nmating, children, it, pop, hall_of_fame):
        """
        add children to population, then do reverse elitism (discard worse) on all
        """
        if self.replace:
            previous = (pop.get_best())
        #print(children.get_size())
        pop.merge(children)
        if self.hof != 0.:
            pop.merge(hall_of_fame)
            nhof = hall_of_fame.get_size()
            #if self.debug:
            if self.debug: print("hall of fame size in FitSelection ",nhof,pop.get_size(),nmating)
        else:
            nhof = 0.   
        pop.sortpop(self.fhigh)
        if self.savechrm:
            size = pop.get_size()
            self.history.spawn(nmating,pop.chromosomes[size-nmating:],pop.fitness[size-nmating:])
        pop.kill(int(nmating+nhof))
        if self.replace:
            now = (pop.get_best())
            if previous[0] < now[0] and self.fhigh == False:
                    pop.fitness[0]     = previous[0]
                    pop.chromosomes[0] = previous[1]
                    if pop.coding_only == False:
                        pop.specimens[0]   = specimens[pop.previous[2]]
            elif previous[0] > now[0] and self.fhigh == True:
                    pop.fitness[0]     = previous[0]
                    pop.chromosomes[0] = previous[1]
                    if pop.coding_only == False:
                        pop.specimens[0]   = specimens[pop.previous[2]]            
        best, bestchrm, bestID = pop.get_best()
        return best, bestchrm, bestID
