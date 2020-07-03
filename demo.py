# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 14:16:24 2020

@author: Amin
"""

from visualizations import visualize_average_segmentation
from BayesianGMM.BayesianGMM import BayesianGMM
from DataHandling import DataHandler
import numpy as np
from BUtils import Simulator


config = {'random_init':True, # If True, locations and colors are initialzed randomly
          'do_update_sigma': True, # If True, cell shapes get updated through iterations
          'gt_sigma': False} # Setting the sigma to their ground truth value

# %% Load the statistical atlas and choose a subset of cells
file = 'atlas.mat'
atlas = DataHandler.load_atlas(file,'tail')

neurons = ['PDA', 'DVB', 'PHAL', 'ALNL', 'PLML']

indices = np.array([atlas['names'].index(neuron) for neuron in neurons])
atlas['names'] = neurons
atlas['mu'] = atlas['mu'][indices,:]
atlas['sigma'] = atlas['sigma'][:,:,indices]

atlas['mu'][:,:3] = atlas['mu'][:,:3] - atlas['mu'][:,:3].min(0) + 5


# %% Generative data hyperparameters
params = {'B':0, # Number of background components (always set to zero)
          'N':5000, # Number of observed pixels
          'Σ_loc':1, # Hyperparameter for generating neuronal shapes (controls mean size)
          'Σ_scale':.1 # Hyperparameter for generating neuronal shapes (controls the )
          }

gt = Simulator.simulate_gmm(atlas,N=params['N'],B=params['B'],Σ_loc=params['Σ_loc'],Σ_scale=params['Σ_scale'])

n_trials = 1 # Number of trials
n_iter   = 1000 # Max number of iterations (each algorithm is run for 1 second)

# %% Run algorithms 
sems = []
vems = []
oems = []

for trial in range(n_trials):    
    oem = BayesianGMM(atlas, gt['X'].numpy(), noise_props=np.empty(0),random_init=config['random_init'])
    oem.do_sinkhorn     = False
    oem.do_update_pi    = True
    oem.do_update_sigma = config['do_update_sigma']
    if config['gt_sigma']:
        oem.sigma = gt['Σ'].permute((1,2,0)).numpy()
    
    oacc = []
    
    
    vem = BayesianGMM(atlas, gt['X'].numpy(), noise_props=np.empty(0),random_init=config['random_init'])
    vem.do_sinkhorn     = False
    vem.do_update_pi    = False
    vem.do_update_sigma = config['do_update_sigma']
    if config['gt_sigma']:
        vem.sigma = gt['Σ'].permute((1,2,0)).numpy()
    
    vacc = []
    
    sem = BayesianGMM(atlas, gt['X'].numpy(), noise_props=np.empty(0),random_init=config['random_init'])
    sem.do_sinkhorn     = True
    sem.do_update_pi    = False
    sem.do_update_sigma = config['do_update_sigma']
    if config['gt_sigma']:
        sem.sigma = gt['Σ'].permute((1,2,0)).numpy()
    
    sacc = []
    
    for iter in range(n_iter):
        if oem.timing.sum() <= BayesianGMM.max_time:
            oem.iterate()
            oacc.append(oem.compute_accuracy(atlas['names'], gt['µ'].numpy()[:,:3], 1, radius=1))
        
        if vem.timing.sum() <= BayesianGMM.max_time:
            vem.iterate()
            vacc.append(vem.compute_accuracy(atlas['names'], gt['µ'].numpy()[:,:3], 1, radius=1))
        
        if sem.timing.sum() <= BayesianGMM.max_time:
            sem.iterate()
            sacc.append(sem.compute_accuracy(atlas['names'], gt['µ'].numpy()[:,:3], 1, radius=1))
            
        if oem.timing.sum() > BayesianGMM.max_time and\
           vem.timing.sum() > BayesianGMM.max_time and\
           sem.timing.sum() > BayesianGMM.max_time:
               
            break
        
    oem.acc = oacc
    vem.acc = vacc
    sem.acc = sacc
    
    oem.gt = {'mu':gt['µ'].data.numpy(), 'sigma':gt['Σ'].data.numpy(), 'pi':gt['π'].data.numpy(), 'Z':gt['Z'].data.numpy()}
    vem.gt = {'mu':gt['µ'].data.numpy(), 'sigma':gt['Σ'].data.numpy(), 'pi':gt['π'].data.numpy(), 'Z':gt['Z'].data.numpy()}
    sem.gt = {'mu':gt['µ'].data.numpy(), 'sigma':gt['Σ'].data.numpy(), 'pi':gt['π'].data.numpy(), 'Z':gt['Z'].data.numpy()}
    
    oems.append(oem)
    vems.append(vem)
    sems.append(sem)
    
ems = [oems, vems, sems]
labels = ['oEM', 'vEM', 'sEM']

visualize_average_segmentation(gt['X'],gt['µ'],gt['Σ'],ems,labels,save=False)
