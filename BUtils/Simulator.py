# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 14:13:09 2020

@author: Amin
"""
import pyro.distributions as dist
import numpy as np
import torch
import pyro


def simulate_gmm(atlas,N=5000,Σ_loc=1,Σ_scale=.1,B=0,noise_props=[[]],noise_colors=[[],[],[]],bounds=None):
    
    C       = atlas['mu'].shape[1]-3 # Number of colors
    K       = atlas['mu'].shape[0] # Number of components

    
    noise_props = torch.tensor(noise_props).float()
    noise_colors = torch.tensor(noise_colors).float()
    if bounds is not None:
        bounds = torch.tensor(bounds).float()
    
    π       = torch.cat(((1-noise_props.sum())*torch.ones((1,K)).float()/K,noise_props),1) # Mixing weights
    
    cov = np.zeros(((C+3)*(K),(C+3)*(K)))
    for n in range(K):
        cov[6*n:6*n+6,6*n:6*n+6] = atlas['sigma'][:,:,n]
        
    μ_p = torch.tensor(atlas['mu']).float()
    Σ_p = torch.tensor(cov).float()
    
    Σ_loc = torch.tensor(Σ_loc).float().repeat(K+B)
    Σ_scale = torch.tensor(Σ_scale).float().repeat(K+B)
    
    Σ_loc[K:K+B] = 1
    Σ_scale[K:K+B] = 1
    
    # %% Sample generative data
    µ = torch.zeros((K+B,C+3))
    µ[:K,:] = pyro.sample('µ', dist.MultivariateNormal(µ_p.reshape(-1), Σ_p)).view(µ_p.shape)
    if B > 0:
        µ[K:K+B,:3] = pyro.sample('µ', dist.Uniform(torch.tensor([0,0,0]).float(),bounds).expand([B,B]))
        µ[K:K+B,3:] = pyro.sample('µ', dist.MultivariateNormal(noise_colors.reshape(-1), torch.eye(noise_colors.numel()))).view(noise_colors.shape)
    
    with pyro.plate('components', K+B) as ind:
        Σ = pyro.sample('Σ', dist.LogNormal(Σ_loc[ind],Σ_scale[ind]))[:,np.newaxis,np.newaxis]*torch.eye(C+3)[np.newaxis,:,:]
    
    with pyro.plate('data', N):
        Z = pyro.sample('Z', dist.Categorical(π))
        X = pyro.sample('X', dist.MultivariateNormal(µ[Z], Σ[Z]))
    
    return {'X':X, 'Z':Z, 'π':π, 'µ':µ, 'Σ':Σ}

