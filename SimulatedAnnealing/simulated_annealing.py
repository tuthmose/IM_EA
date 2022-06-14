import numpy as np
import scipy as sp
import sys

# Giordano Mancini Jan 2019

def get_func_from_name(name):
    """
    given function and module names, return function
    """
    modn, funcn = name.rsplit('.', 1)
    if modn not in sys.modules:
        __import__(modn)
    func = getattr(sys.modules[modn],funcn)
    return func

def d1_uf(x,f):
    """
    Finite difference for f'
    using numpy arrays
    """
    n = x.shape[0]
    #h = x[1]-x[0]
    h = np.mean(x[1:] - x[0:-1])
    left  = f[:-1]
    right = f[1:]
    last = np.array((right[-1]-right[-2]))
    return np.append(np.subtract(right,left),last)/h
  
class SimulatedAnnealing(object):
    
    def __init__(self,**kwargs):
        """
        instatiate SA; keywords contain defaults
        objfunc : objective function (mandatory)
        genfunc : hot to generate new states (default: generate new conf from domain)
        cutoff  : for generating near new configurations
        conf0   : starting configuration (mandatory)
        domain  : inclusive domain extent for data (default [0,1])
        cooling_scheme: if None, just use a linear constant (see c. rate)
                        otherwise it is a custom function with kwd arguments
                        (temp, move_reg (see below), Energies)
        cooling_rate : slope for linear cooling; if=-1 do a simple MC search
        min_steps    : minimum number of steps for slope_cooling
        lower_fail   : wether to lower temperature if move was rejected (default: True)
        Econv  : target energy, if any   (default: None)
        tol    : convergence tolerance (default 1e-8)
        gkwd, okwd, ckwd : keywords for genfunc, objfunc and cooling_scheme
        """
        # check defaults
        prop_defaults = {
            "objfunc" : None, 
            "genfunc" : newconf_global,
            "cutoff"  : None,
            "conf0"   : None,            
            "domain"  : (0,1),
            "cooling_scheme": linear_cooling,
            "cooling_rate"  : 0.995,
            "lower_fail"    : True,
            "min_steps"     : 20,
            "Econv"   : None,
            "tol"     : 1e-8,
            "gkwd"    : None,
            "okwd"    : None,
            "ckwd"    : None
        }
        for (prop, default) in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))
        ## setup
        if self.objfunc == None:
            raise ValueError("Missing objective function")
        if isinstance(self.conf0,np.ndarray):
            if not np.any(self.conf0):
                raise ValueError("Missing start configuration")
        elif self.conf0 == None:
            raise ValueError("Missing start configuration")
        if self.cooling_rate == -1.:
            self.cooling_rate = 1.
        else:
            assert self.cooling_rate > 0. and self.cooling_rate < 1.
            
        if self.gkwd is not None:
            self.gkwd = dict(self.gkwd)
        else:
            self.gkwd = dict()
        if self.okwd is not None:
            self.okwd = dict(self.okwd)
        else:
            self.okwd = dict()
        if self.ckwd is not None:
            self.ckwd = dict(self.ckwd)
        else:
            self.ckwd = dict()            
        #self.gkwd['domain'] = self.domain
        #self.gkwd['cutoff'] = self.cutoff            

    def Anneal(self,**kwargs):
        """
        tstart : starting temp   (default: 1000.0)
        tmin   : minimum temperature (for non constant scaling schemes)
        nsteps : number of steps (default: 1000)
        states : where to append accepted moves 
                 default None; in chem. appl. invoking the wrapper automatically
                 saves data
                 otherwise states is a filename where to write a numpy array of 
                 coordinates
        Boltzmann  : Boltzman constant
        """
        # check defaults
        prop_defaults = {
            "tstart"    : 1000.,
            "tmin"      : 1.,
            "nsteps"    : 5000, 
            "states"    : None,
            "verbose"   : False,
            "Boltzmann" : 1.,
            "restart"   : 0,
            "seed"      : None
            }
        for (prop, default) in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))
        if self.states != None:
            if not isistance(str,self.states):
                raise ValueError("Provide a valid filename for states")
        if self.seed is not None:
            np.random.seed(self.seed)
        self.traj = list()
        self.Energies  = list()
        if self.lower_fail == True and self.cooling_rate < 1.:
            ft = self.tstart
            for i in range(self.nsteps):
                ft = ft*self.cooling_rate
                ft = max(self.tmin,ft)
            print("+++ Final temperature will be {0:f} with linear cooling and lower_fail=True".format(ft))
        elif self.cooling_rate == 1.:
            print("+++ Cooling rate = 1: plain MC search")

        Energy = self.objfunc(self.conf0,**self.okwd)
        if len(Energy) > 1:
               Energy = Energy[0]
        print("+++ Starting Energy is ",Energy)
        
        conf = self.conf0
        Temp = self.tstart
        if self.restart > 0:
            if self.cooling_rate < 1.:
                raise ValueError("Restart only with plain MC")
            else:
                print("+++ Restart from worst after every ",self.restart, " failed moves")
        failed = 0
        
        #--------- main loop
        for it in range(self.nsteps):
            # gen and evaluate candidate
            newconf  = self.genfunc(conf,**self.gkwd)
            newE     = self.objfunc(newconf,**self.okwd)
            if len(newE) > 1:
               newE = newE[0]            
            accepted, how = self.evaluate(Energy,newE,Temp)
            if accepted:
                #save new data
                Energy = newE
                self.Energies.append([it,int(how),Temp,Energy])
                conf = newconf
                self.traj.append(np.append(conf,newE))
                failed = 0
            else:
                failed =+ 1
                if failed >= self.restart and self.restart > 0:
                    failed = 0
                    wconf, wE = self.get_worst()
                    if wconf != None:
                        print("+++ restarting from Energy= ",wE)
                        conf = wconf
                        Energy = wE
            if self.Econv != None:
                if abs(Energy - Econv) <= tol:
                    print("+++ converged to target {0:f} at step {1:d}".format(abs(Energy-Econv),it))
                    break
            if self.verbose >= 2:
                print(it, Energy, len(self.Energies),failed, self.restart)
            #new temp
            if self.cooling_rate < 1.:
                Temp = self.gen_temp(accepted,Temp,self.cooling_scheme,optargs=self.ckwd)
        #------- end main loop
        
        #save traj
        if self.states != None:
            self.traj = np.asarray(self.traj)
            np.savetxt(self.states,self.traj)
            
        # get best conf
        self.Energies = np.asarray(self.Energies)
        best = np.argmin(self.Energies[:,3])
        acc_ratio = float(self.Energies.shape[0])/float(self.nsteps)
        return self.Energies[best][0], self.Energies[best][3], Temp, acc_ratio
      
    def evaluate(self,Eold,Enew,Temp):
        if Enew <= Eold:
            return True, True
        Ediff   = (Enew-Eold)/(self.Boltzmann * Temp)
        Bweight = np.exp(-Ediff)
        coin = np.random.rand()
        if Bweight > coin:
            return True, False
        else:
            return False, False
        
    def gen_temp(self,accepted,Temp,func,**kwargs):
        if self.cooling_rate > -1. and (accepted or self.lower_fail):
            if Temp <= self.tmin:
                return self.tmin
            else:
                kwargs['cooling_rate'] = self.cooling_rate
                kwargs['Energies'] = self.Energies
                kwargs['min_steps'] = self.min_steps
                return func(Temp,**kwargs)
        else:
            return Temp

    def get_worst(self):
        """
        get worst (highest energy) configuration
        sampled until now and saved in self.traj
        if no moves have been accepted, just 
        return None
        """
        if len(self.Energies) == 0:
            return None, None
        wE = np.argsort(self.Energies[:,2])
        wConf = self.traj[wE]
        assert self.Energies[wE,2] == self.traj[wE,-1]
        return self.traj[wE,:-1], self.traj[wE,-1]

def linear_cooling(Temp,**kwargs):
    """
    simple linear cooling scheme with given constant
    """
    cooling_rate = kwargs.get('cooling_rate')
    return cooling_rate*Temp

def slope_cooling(Temp,**kwargs):
    """
    calculate cooling constant as derivative of Energies
    vs number of moves for the last few steps
    if less than the minimum steps have been carried out
    use linear cooling
    """  
    cooling_rate = kwargs.get('cooling_rate')
    Energies     = kwargs.get('Energies')
    min_steps    = kwargs.get('min_steps')
    Y = np.array(Energies)
    nx = Y.shape[0]
    if nx <= min_steps:
        return cooling_rate*Temp
    slope = d1_uf(np.arange(min_steps),Y[nx-min_steps:,2])
    variable_rate = np.abs( np.mean( slope/np.max(np.abs(slope)) ))/10.
    Temp = (variable_rate+0.9)*Temp
    return Temp
    
def newconf_global(coords,**kwargs):
    """
    generate a new set of coordinates from the complete domain
    """
    domain = kwargs.get('domain')
    nvar = coords.shape
    newcoords = np.random.random_sample(nvar)
    dist = domain[1]-domain[0]
    newcoords *= dist
    newcoords += domain[0]
    return newcoords

def newconf_near(coords,**kwargs):
    """
    generate a new conf in the neighbourhood 
    of the input
    cutoff is either a single value or has the same
    shape of conf
    """   
    domain = kwargs.get('domain')
    cutoff = kwargs.get('cutoff')
    if not isinstance(cutoff,np.ndarray):
        cutoff = cutoff*np.ones(coords.shape[0])
    newcoords = np.empty(coords.shape)
    try:
        for i, value in enumerate(cutoff):
            newcoords[i] = coords[i] + 2.*value*np.random.rand()-value
    except TypeError:
        assert cutoff.shape == coords.shape
        newcoords += coords + (2.*np.random.random_sample(cutoff.shape)-1)*cutoff
    return newcoords
