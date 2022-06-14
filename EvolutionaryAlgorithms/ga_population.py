import numpy as np
import ga_utils
import cluster_utils
from scipy.spatial.distance import cdist,pdist

# Giordano Mancini Dec 2018
### Population related classes 
                     
class GA_population:
    def __init__(self,**kwargs):
        """
        Given numpy arrays for genetic and non-genetic features creates structures
        for GA optimization. Data is stored as two or three numpy arrays holding:
            - chromosomes: data used by genetic operators
            - fitness
        and an optional list:
            - specimens: all other information needed by fitness evaluator 
            (e. g. a collection of zmat)
            specimen(s) must have the following methods:
                - get/set fitness
                - get/set chromosomes
                - a list of selected features to be used as chromosomes
        in addition, a template file must be provided to create new specimens
        the specimen type may be provided with sptype (if coding_only==False)
        otherwise defaults to "mol" (ga_utils.specimen)
        chromosomes are indexed by their position and all data is sorted by fitness 
        """
        pop_defaults = {
            "chromosomes"  : np.empty(0),
            "fitness"      : np.empty(0),
            "specimens"    : list(),
            "template"     : None,
            "genes"        : None,
            "coding_only"  : False,
            "ngene"        : 0,
            "sptype"       : "mol",
            "caller_type"  : None
        }
        for (prop, default) in pop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))
        # done reading options
        size = self.get_size()
        if np.any(self.fitness):
            assert size == self.fitness.shape[0]
        if len(self.chromosomes) == 0:
           if hasattr(self.genes, '__iter__') and len(self.genes) > 0 and self.coding_only is False:
            self.ngene = len(self.genes)
        elif self.chromosomes[0].dtype == 'U1' and len(self.chromosomes) > 0:
            self.ngene = self.chromosomes.shape[1]
        elif np.any(self.chromosomes):
                self.ngene = self.chromosomes.shape[1]
        elif self.coding_only:
            self.ngene = 0
        else:
            raise ValueError("With non coding data provide either genes or chromosomes")
        if self.coding_only is False:
            if np.any(self.chromosomes):
                assert len(self.specimens) == size
        if len(self.specimens) > 0:
            self.coding_only = False
            
    def insert_specimen(self,chrm,sp,newfit=False):
        if sp != None:
            self.specimens.append(sp)
        if self.get_size() > 0:
            newsize = self.get_size() + 1
            newshape = self.get_shape(newsize,chrm.shape)
            self.chromosomes = np.append(self.chromosomes,chrm)
            self.chromosomes.shape = newshape
        else:
            self.chromosomes = chrm
        if np.any(newfit) and self.fitness.size != 0:
            self.fitness = np.hstack((self.fitness,newfit))
        elif np.any(newfit):
            self.fitness = newfit
        else:
            f = np.zeros(1)
            self.fitness = np.hstack((self.fitness,f))
        
                        
    def get_size(self):
        if self.chromosomes.ndim == 1:
            if len(self.chromosomes) > 0:
                return 1
            else:
                return 0
        else:
            return  self.chromosomes.shape[0]
    
    def get_shape(self,num,shape2):
        if self.chromosomes.ndim == 1 and len(shape2)>1:
            # one chrm but more than one new
            newshape = list(shape2)
            newshape[0] = num
        elif self.chromosomes.ndim == 1 and len(shape2)==1:
            # one new, one old
            newshape = [2,shape2[0]]
        elif self.chromosomes.ndim > 1:
            # more than one chromosome;
            if len(self.chromosomes.shape) == len(shape2):
                newshape = list(self.chromosomes.shape)
                newshape[0] = num
            else:
                newshape = list(self.chromosomes.shape)
                newshape[0] = num
        return tuple(newshape)

    def spawn(self, num, newchrm, newfit=False):
        """
        expand chromosomes, fitness and specimens with new data
        """
        newchrm = np.asarray(newchrm)
        newfit  = np.asarray(newfit)
#       if self.ngene == 0:
#           if newchrm.ndim > 1:
#               self.ngene = newchrm.shape[1]
#           else:
#               self.ngene = newchrm.shape[0]
#       print(newchrm.shape)
#        if self.chromosomes.size != 0:
        if self.get_size() > 0:
            newsize = self.get_size() + num
            newshape = self.get_shape(newsize,newchrm.shape)
            self.chromosomes = np.append(self.chromosomes,newchrm)
            self.chromosomes.shape = newshape
        else:
            self.chromosomes = newchrm
        # make sense only if chrm data is 1D
        if self.ngene == 0 and len(self.chromosomes.shape)<3:
            if self.get_size() > 1:
                self.ngene = self.chromosomes.shape[1]
        if np.any(newfit) and self.fitness.size != 0:
            self.fitness = np.hstack((self.fitness,newfit))
        elif np.any(newfit):
            self.fitness = newfit
        else:
            f = np.zeros(num)
            self.fitness = np.hstack((self.fitness,f))          
        if self.coding_only is False:
            if num == 1:
                if self.sptype == 'cluster' and self.caller_type == 'xTB':
                    newsp = cluster_utils.xtb_cluster(newchrm, self.template, self.genes)
                elif self.sptype == 'cluster' and self.caller_type == 'g16':
                    newsp = cluster_utils.gau_cluster(newchrm, self.template, self.genes)
                else:
                    newsp = ga_utils.linear_molecule(self.template,self.genes,newchrm)
                self.specimens.append(newsp)
            else:
                for nc in range(num):
                    #if self.specimens[0].__class__.__name__ == 'cluster':
                    if self.sptype == 'cluster' and self.caller_type == 'xTB':
                        newsp = cluster_utils.xtb_cluster(newchrm[nc], self.template, self.genes)
                    elif self.sptype == 'cluster' and self.caller_type == 'g16':
                        newsp = cluster_utils.gau_cluster(newchrm[nc], self.template, self.genes)                       
                    else:
                        newsp = ga_utils.linear_molecule(self.template, self.genes, alleles=newchrm[nc])
                    self.specimens.append(newsp)
       
    def savepop(self,prefix):
        """
        save to prefix_fitness and prefix_chromosomes
        current population and return it
        specimens are not saved
        """
        fname = prefix + "_fitness.dat"
        np.savetxt(fname,self.fitness)
        cname = prefix + "_chromosomes.dat"
        np.savetxt(cname,self.chromosomes)
        return self.fitness, self.chromosomes
        
    def kill(self,tokill):
        """
        kill chromosomes; if to kill is an int eliminate last n 
        otherwise, use a list
        """
        if isinstance(tokill,int):
            size = self.get_size()
            self.chromosomes = self.chromosomes[:size-tokill]
            self.fitness     = self.fitness[:size-tokill]
            if self.coding_only is False:
                self.specimens    = self.specimens[:size-tokill]
        elif isinstance(tokill,list) or isinstance(tokill,tuple):
            self.chromosomes = np.delete(self.chromosomes,tokill,axis=0)
            self.fitness     = np.delete(self.fitness,tokill)
            if self.coding_only is False:
                tmp = list()
                for i,s in enumerate(self.specimens):
                    if i in tokill:
                        continue
                    else:
                        tmp.append(s)
                del self.specimens
                self.specimens = tmp
                
    def calc_fitness(self,ffunc,n,fitkwds):
        """
        calculate fitness defined in ffunc on last n chromosomes
        or on input given list
        """
        if fitkwds == None:
            fitkwds = dict()
        if self.coding_only:
            #if self.get_size() == 1:
            #    self.chromosomes = np.expand_dims(self.chromosomes,axis=0)
            if isinstance(n,int):
                s = self.get_size()
                self.fitness = np.apply_along_axis(ffunc,1,self.chromosomes[s-n:],**fitkwds)
            elif isinstance(n,list) and len(n) > 0:
                self.fitness[n] = np.apply_along_axis(ffunc,1,self.chromosomes[n],**fitkwds)
        else:
            if isinstance(n,int):
                s = self.get_size()
                for i in range(s-n,s):
                    # fitness and chromosomes in specimens updated by calling ffunc
                    f, c = ffunc(self.specimens[i],**fitkwds)
                    if c is not None:
                        self.chromosomes[i] = c
                    self.fitness[i] = f
            elif isinstance(n,list):
                for i in n:
                    # fitness and chromosomes in specimens updated by calling ffunc
                    f, c = ffunc(self.specimens[i],**fitkwds)
                    if c is not None:
                        self.chromosomes[i] = c
                    self.fitness[i] = f
                    
    def sortpop(self,best_highest):
        """
        sort population using fitness in ascending (best=False)
        or descending order
        """
        order = np.argsort(self.fitness)
        if best_highest is True:
            order = order[::-1]
        self.fitness     = self.fitness[order]
        self.chromosomes = self.chromosomes[order]
        if self.coding_only is False:
            tmp = [self.specimens[x] for x in order]
            self.specimens = tmp
        
    def get_one(self,n):
        """
        return chromosome n
        """
        return self.chromosomes[n]
    
    def get_one_sp(self,n):
        """
        return specimen n
        """
        return self.specimens[n]
        
    def get_best(self):
        """
        return best current fitness
        assumes sorted chromosomes
        """
        if self.coding_only:
            return self.fitness[0], self.chromosomes[0], None
        else:
            return self.fitness[0], self.chromosomes[0], self.specimens[0].ID
        
    def get_ch_labels(self):
        """
        return indexes or labels of chromosomes
        """
        return list(range(self.chromosomes.shape[0]))
        
    def mutate_specimen(self,chrm,newallele):
        """
        set random gene in mutated chromosomes / specimens
        to newallele
        """
        gmut = np.random.choice(self.ngene)
        self.chromosomes[chrm,gmut] = newallele
        if self.coding_only is False:
            self.specimens[chrm].set_chromosome([newallele],[gmut])
                   
    def merge(self,pop):
        """
        join two populations
        """
        self.chromosomes = np.vstack((self.chromosomes,pop.chromosomes))
        self.fitness = np.hstack((self.fitness,pop.fitness))
        if self.coding_only is False:
            self.specimens = self.specimens + pop.specimens

