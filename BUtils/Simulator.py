# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 14:13:09 2020

@author: Amin
"""
import torch
import pyro
import pyro.distributions as dist
import numpy as np

def simulate_gmm(atlas,N=5000,B=0,Σ_loc=1,Σ_scale=.1):
    
    C       = atlas['mu'].shape[1]-3 # Number of colors
    K       = atlas['mu'].shape[0] # Number of components


    π       = torch.ones((1,K)).float()/K # Mixing weights
    
    cov = np.zeros(((C+3)*(K+B),(C+3)*(K+B))) 
    for n in range(K):
        cov[6*n:6*n+6,6*n:6*n+6] = atlas['sigma'][:,:,n]
    
    μ_p = torch.tensor(atlas['mu']).float()
    Σ_p = torch.tensor(cov).float()
    
    
    
    # %% Sample generative data
    µ = pyro.sample('µ', dist.MultivariateNormal(µ_p.reshape(-1), Σ_p)).view(µ_p.shape)
    
    with pyro.plate('components', K+B):
        Σ = pyro.sample('Σ', dist.LogNormal(Σ_loc,Σ_scale))[:,np.newaxis,np.newaxis]*torch.eye(C+3)[np.newaxis,:,:]
    
    with pyro.plate('components', K+B):
        Σ = pyro.sample('Σ', dist.LogNormal(Σ_loc,Σ_scale))[:,np.newaxis,np.newaxis]*torch.eye(C+3)[np.newaxis,:,:]
        
    with pyro.plate('data', N):
        Z = pyro.sample('Z', dist.Categorical(π))
        X = pyro.sample('X', dist.MultivariateNormal(µ[Z], Σ[Z]))
    
    return {'X':X, 'Z':Z, 'π':π, 'µ':µ, 'Σ':Σ}