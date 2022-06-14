Conformer search with different metaheuristics and wrappers to Electronic
structure codes.

Both *Simulated annealing*  and *Evolutionary algorithm* codes can be used to
optimize arbitrary cost functions (which must be provided) and can operate
on numpy arrays or arbitrary objects called *specimens*. In the latter case,
genetic operations are still done on *chromosomes*; modifications to *chromosomes*
are transmitted back and forth to *specimens* via the custom fitness function.

For the esploration of conformer space, the search can be made on rotatable bonds
(with or without local convex optimization in G16) (*zmat_specimen*) or directly
on cartesian coordinates (*cluster_specimen*; experimental).

- `molecule_utils` include a parser for Gaussian16 and a `zmatrix` module for internal
  coordinate manipulation and z-matrix/cartesian conversion. 
- `EvolutionaryAlgorithms` contains `ga_population` and `ga_evolution` which implements
   an island $\lambda,\mu$ model.
- `SimulatedAnnealing` implements a simple SA algorithm; custom functions for the cooling
  scheme and generation of neighbours can be provided.
- `test` containts a few examples jupyter notebook and shell scripts with submission and post 
    processing pipelines

please cite one or more of the following studies:
- (1) Mancini, G.; Fusè, M.; Lipparini, F.; Nottoli, M.; Scalmani, G.; Barone, V. Molecular Dynamics Simulations Enforcing Nonperiodic Boundary Conditions: New Developments and Application to the Solvent Shifts of Nitroxide Magnetic Parameters. J. Chem. Theory Comput. 2022, acs.jctc.2c00046. https://doi.org/10.1021/acs.jctc.2c00046.
(2) Mancini, G.; Fusè, M.; Lazzari, F.; Chandramouli, B.; Barone, V. Unsupervised Search of Low-Lying Conformers with Spectroscopic Accuracy: A Two-Step Algorithm Rooted into the Island Model Evolutionary Algorithm. J. Chem. Phys. 2020, 153 (12), 124110. https://doi.org/10.1063/5.0018314.
(3)  Mancini, G.; Del Galdo, S.; Chandramouli, B.; Pagliai, M.; Barone, V. Computational Spectroscopy in Solution by Integration of Variational and Perturbative Approaches on Top of Clusterized Molecular Dynamics. J. Chem. Theory Comput. 2020, 16 (9), 5747–5761. https://doi.org/10.1021/acs.jctc.0c00454.

