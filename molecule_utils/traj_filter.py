import math
import numpy as np
import scipy as sp
from collections import deque
from scipy import constants
from rotconstant import gettrajB
import MDAnalysis.analysis.rms as MDArms 

kB = 0.001*sp.constants.physical_constants['Boltzmann constant'][0]\
*constants.physical_constants['Avogadro constant'][0]

har2kjmol = sp.constants.physical_constants["Avogadro constant"][0]\
*sp.constants.physical_constants["Hartree energy"][0]/1000.

def find_centroid(D, beta=1., mask=None):
    """
    find centroid on input distance matrix using similarity scores:
    s_ij = e^-beta rmsd_ij / sigma(rmsd)
    c = argmax_i sum_j sij
    i.e. the frame with maximum similarity from all the others
    """
    similarity = np.exp(-beta*D/np.std(D))
    if mask is None:
        centroid = (similarity.sum(axis=1)).argmax()
    else:
        maskt = np.expand_dims(mask,axis=0).T
        sim_m = np.ma.array(similarity,mask=~(mask*maskt))
        centroid = (sim_m.sum(axis=1)).argmax()
    return centroid  

def find_centroid_simple(D, mask=None):
    if mask is None:
        centroid = (D.sum(axis=1)).argmin()
    else:
        maskt = np.expand_dims(mask,axis=0).T
        d_m = np.ma.array(D, mask=~(mask*maskt))
        centroid = (d_m.sum(axis=1)).argmax()
    return centroid

class sparsify_traj:
    """
    Filter a trajectory for the most unique (RMSD based) and relevant
    (energy based) structures.
    
    1. Add global GEM and centroid to empty selection
    2. build RMSD graph
    3. remove all edges with rmsd > cutoff
    4. identify disconnected regions
    5. in each region with more than one node
        select centroid and local energy minima
    6. add singles
    """
    
    def __init__(self,**kwargs):
        """
        rmsd and energy are mandatory arguments,
        nselect is at least 2 (centroid and 
        energy minimum)
        nselect 3 adds point farthest from centroid
        nselect 4 adds point farthest from em
        """
        prop_defaults = {
            "rmsd"      : None,
            "energy"    : None,
            "nselect"   : 2,
            "verbose"   : 0
        }
        for (prop, default) in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))        
        assert isinstance(self.rmsd,np.ndarray)
        assert isinstance(self.energy,np.ndarray)
        assert self.rmsd.shape[0] == self.rmsd.shape[1]
        assert self.energy.shape[0] == self.rmsd.shape[1]
        assert isinstance(self.nselect, int)
        self.nframe = self.rmsd.shape[1]
        
    def run(self, **kwargs):
        """
        cutoff : RMSD cutoff in nm
        """
        prop_defaults = {
           "cutoff" : 0.005,
            "save"  : True
        }
        for (prop, default) in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))
        assert isinstance(self.cutoff, float)
        # step 1
        selected = list()
        GEM = np.argmin(self.energy)
        global_centroid = find_centroid(self.rmsd)
        selected.append(GEM)
        selected.append(global_centroid)
        # 2,3 
        G = self.build_graph()
        # 4
        clusters, singles = self.find_clusters(G)
        r = list()
        icl = 0
        #for c in clusters:
        #    r = r+c
        #    icl = icl+len(c)
        #print(len(set(r)),len(r),len(clusters))
        #print(len(set(r))+len(singles),icl+len(singles),self.rmsd.shape[0])
        # 5
        representative = self.select_nodes(G, clusters, singles)
        selected = selected + representative + singles
        # 6
        if self.verbose >= 1:
            sizes = [len(c) for c in clusters]
            print("--- sparsify: cluster mean, median and std of size: ",np.mean(sizes),\
                np.median(sizes),np.std(sizes))
            print("--- sparsify: cluster max size ",np.max(sizes))
            print("--- sparsify: found ",len(representative)," points in ",\
                len(clusters)," clusters")
            print("--- sparsify: found ",len(singles)," singles")
            dmax  = np.max(self.rmsd[selected])
            dcmax = np.max(self.rmsd[representative])
            dmin  = np.min(self.rmsd[selected])
            dcmin = np.min(self.rmsd[representative])
            print("--- sparsify: points separation from ",dmin," to ", dmax)
            print("---     ",dcmin,dcmax," excluding singles")
            print("--- sparsify: selected ",len(selected), "points out of ",\
                self.rmsd.shape[0])
        if self.save:
            self.selected = selected
            self.singles  = singles
            self.clusters = clusters
        return selected        
                   
    def build_graph(self):
        """
        Sparsify RMSD matrix above cut off
        """
        G = np.copy(self.rmsd)
        # sparsify
        G[G>self.cutoff] = 0.
        return G
    
    def find_clusters(self, G):
        """
        return a list of lists of connected nodes
        """
        clusters = list()
        nodes = np.arange(self.nframe,dtype='int')
        # PHASE 1: BFS of G
        # start with first element
        first = 0
        n_vis = 0
        while True:
            visited, deck = set(), [first]
            while deck:
                #add current structure to visited if wasn't there
                #list and remove
                # from queue
                node = deck.pop()
                if node not in visited:
                    visited.add(node)
                    # all neighbours of vertex
                    deck = deck + list(np.where(G[node]>0)[0])
            visited = list(visited)
            clusters.append(visited)
            nodes[visited] = -1
            n_vis = n_vis + len(visited)
            if n_vis == self.nframe:
                break
            elif n_vis > self.nframe:
                raise ValueError("more visited nodes than frames")
            #first available
            first = nodes[nodes!=-1][0]
        # CLUSTERING DONE
        # PHASE 2: sort clusters on their size and put 
        # singletons in a different data str.
        # split in singles and real clusters
        # could calculate a MST to rank clusters
        singles = list()
        clusters = sorted(clusters,key=lambda x: len(x),reverse=True)
        for i in range(len(clusters)):
            if len(clusters[i])==1:
                break
        for j in range(i,len(clusters)):
            singles.append(clusters.pop().pop())       
        return clusters, singles
    
    def select_nodes(self, G, clusters, singles):
        N = np.arange(self.nframe,dtype='int')
        representative = list()
        # pick energies and distances of cluster members
        for icl, cl in enumerate(clusters):
            #print(icl,cl,len(cl),len(set(cl)))
            if len(cl) <= 2:
                representative = representative + cl
                continue
            #print(cl)
            cl = np.sort(cl)
            #print(cl)
            mask = np.isin(N,cl)
            E = np.ma.array(self.energy, mask=~mask)
            em  = np.argmin(E)
            local_centroid = find_centroid(self.rmsd, mask=mask)
            #print(local_centroid, em,cl)
            if em in representative or local_centroid in representative:
                print(icl, cl, em, local_centroid, representative)
                raise ValueError
            representative.append(local_centroid)
            if em != local_centroid:
                representative.append(em)
            if self.nselect >= 3:
                maskt = np.expand_dims(mask,axis=0).T
                D = np.ma.array(self.rmsd, mask=~(mask*maskt))
                flocal = np.argmax(D[local_centroid])
                if flocal not in representative[-2:]:
                    representative.append(flocal)
            if self.nselect >= 4:
                fem = np.argmax(D[em])
                if fem not in representative[-3:]:
                    representative.append(fem)
        # check that there are no replicated members
        r = set(representative)
        if not len(r) == len(representative):
            print("len set ", len(r), "len list  ",len(representative))
            raise ValueError("selected points contains duplicates")
        return representative

def crestlike_filtering(dataframe, traj, ethr=15, dethr=0.4, rmsdthr=0.125, rotthr=0.01, atom_indices=None):
    """_summary_

    Args:
        dataframe (pd.DataFrame): pandas Dataframe with energies
        traj (md.traj): mdtraj trajectory object
        ethr (int, optional): Energy treshold, only structures below are considered. Defaults to 15.
        dethr (float, optional): energy difference within similar structures. Defaults to 0.4. (Angstrom)
        rmsdthr (float, optional): RMSD threshold. Defaults to 0.125. (Angstrom)
        rotthr (float, optional): rotational constants threshold. Defaults to 0.01.
        atom_indices (_type_, optional): Atoms to be considered in RMSD calculation. Defaults to None.

    Returns:
        pd.DataFrame: returns a DataFrame with sorted energies and data from the input dataframe, a new column 'oindx' is added containg the mapping between the selected structure and their original indices
    """
    # get atom masses
    atmass = np.array([a.element.mass for a in traj.topology.atoms],dtype=np.float32)
    # Define an original indx
    tmp_indx = list(range(dataframe.shape[0]))
    tmp_df = dataframe.loc[:,['energy']]
    tmp_df['oindx'] = tmp_indx
    # keep below E thr
    tmp_df['energy'] = ((tmp_df['energy'] - tmp_df['energy'].min())*har2kjmol).to_list()
    mask = tmp_df['energy'] < ethr
    tmp_df = tmp_df[mask]
    # No atom mask used in RotConstants
    bvec = gettrajB(traj[mask])
    tmp_df['Ae'] = bvec[:, 0]
    tmp_df['Be'] = bvec[:, 1]
    tmp_df['Ce'] = bvec[:, 2]
    rconst = ['Ae', 'Be', 'Ce']
    # eng sorting
    tmp_df = tmp_df.sort_values('energy')
    skipp = []
    for i in tmp_df.index:
        # print(i)
        if i in skipp:
            continue
        mask = np.abs(tmp_df['energy'].to_numpy() - tmp_df.loc[i, ['energy']].to_numpy()) < dethr
        smldf = tmp_df[mask]
        loc = smldf.index.get_loc(i)
        #print(smldf)
        # mdtraj RMSD gives weird results
        # tmp_rmsd = md.rmsd(traj[smldf['oindx'].to_list()], traj[smldf['oindx'].to_list()], 0,
        #                    atom_indices=atom_indices)*10
        for j, val in enumerate(smldf.index):
            #print(val)
            if val!= i:
                tmp_rmsd = MDArms.rmsd(traj[smldf['oindx']].xyz[loc][atom_indices][0],
                                       traj[smldf['oindx']].xyz[j][atom_indices][0],
                                       superposition=True,
                                       weights=atmass[atom_indices][0])*10
                if tmp_rmsd < rmsdthr:
                    tmp_thrs = smldf.loc[i, rconst].to_numpy()[0] * rotthr
                    tmp_bdiff = np.abs(smldf.loc[val, rconst].to_numpy() -
                            smldf.loc[i, rconst].to_numpy())[0]
                    if ((tmp_bdiff - tmp_thrs) < 0).any():
                    #print(tmp_df.loc[val])
                        tmp_df = tmp_df.drop(val)
                        skipp.append(val)
    return tmp_df
