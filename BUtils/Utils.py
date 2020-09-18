# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 14:08:59 2020

@author: Amin
"""
import numpy as np
import torch


def obs2image(X1,I1,scale=1,min_subtract=False):
    if len(X1) == 0:
        return
    
    if torch.is_tensor(X1):
        X1 = X1.data.numpy()
        I1 = I1.data.numpy()
        
    X1 = X1*scale
    
    X1 = X1.astype(int)
    
    if min_subtract:
        X1 = X1-X1.min(0)
    else:
        X1[X1 < 0] = 0
    
        
    max_coor = X1.max(0)
    
    shape = np.append(max_coor+1,I1.shape[1])
    
    I = I1.reshape(1,-1).squeeze()
    X = np.repeat(X1,I1.shape[1],0).squeeze()
    C = np.tile(np.arange(I1.shape[1]), X1.shape[0]).astype(int).squeeze()
    
    ind = np.ravel_multi_index((X[:,0], X[:,1], X[:,2], C), shape)
    
    recon = np.zeros((np.prod(shape))); 
    recon[ind] = I/I.max(); recon = recon.reshape(shape)
    
    return recon


def image2obs(data,thresh=2,scale=1):
    C = data.shape[3]
    
    n_pixels = np.prod(data.shape[0:3])
    X_in = np.array(np.where(np.ones(data.shape[0:3]))).T
    I_in = data[:,:,:,:C].reshape((n_pixels,C))
    X_in = X_in[I_in.mean(1) > thresh,:]
    I_in = I_in[I_in.mean(1) > thresh,:]
    
    
    obs = np.concatenate((X_in*scale,I_in[:,:C]),1)
    
    return obs