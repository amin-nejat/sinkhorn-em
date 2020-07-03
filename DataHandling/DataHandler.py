# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 19:10:26 2020

"""

from scipy.io import loadmat


def load_neuropal(file):
    content = loadmat(file)
    
    scale = content['info'][0][0][0].T
    rgbw = content['info'][0][0][2]
    
    return content['data'][:,:,:,rgbw-1].squeeze(), scale, content['worm'][0][0][0][0].lower()
    
def load_atlas(file,bodypart):
    content = loadmat(file)
    
    if bodypart == 'head':
        mu = content['atlas'][0][0][0][0][0][0][0,0][0]
        sigma = content['atlas'][0][0][0][0][0][0][0,0][1]
        names = [content['atlas'][0][0][0][0][0][1][i][0][0] for i in range(mu.shape[0])]
    elif bodypart == 'tail':
        mu = content['atlas'][0][0][1][0][0][0][0,0][0]
        sigma = content['atlas'][0][0][1][0][0][0][0,0][1]
        names = [content['atlas'][0][0][1][0][0][1][i][0][0] for i in range(mu.shape[0])]
    
    mu[:,:3] = mu[:,:3]-1 # Matlab to Python
    
    return {'mu':mu, 'sigma':sigma, 'names': names, 'bodypart':bodypart}

