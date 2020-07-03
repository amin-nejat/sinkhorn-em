# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 19:11:25 2020

@author: Amin
"""

#from pyro.contrib.autoguide import AutoDelta, AutoMultivariateNormal, AutoDiagonalNormal, AutoLowRankMultivariateNormal
#from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete, Trace_ELBO
#from torch.distributions import constraints
#from collections import defaultdict
#import pyro.distributions as dist
#from pyro import poutine
#import torch
#import pyro

from ot.bregman import sinkhorn_knopp
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from Methods.ID.BayesianGMM.BUtils import Utils
from . import Helpers
import numpy as np
import scipy
import copy
import time


class BayesianGMM(object):
    max_iter = 10
    sample = 10000
    converge = 1e-10
    eig_bound = np.array([.1,10.1])
    lambda_sinkhorn = 1
    eps = 1e-10
    max_time = 1
        
    def iterate(obj):
        start = time.time()
        
        obj.E()
        obj.M()
        
        end = time.time()
        
        obj.timing = np.hstack((obj.timing,end-start))
        
        obj.check_convergence()
        
    def optimize(obj):
        for iter in range(BayesianGMM.max_iter):
            obj.iterate()
            
            if obj.converged:
                break
            
    def E(obj):
        if obj.do_subsample:
            obj.subsample()
            
        obj.update_gamma()
        obj.log_likelihood()
        
        if obj.do_sinkhorn:
            obj.update_gamma_sinkhorn()
            
    def M(obj):
        if obj.do_update_pi:
            obj.update_pi()
            
        if obj.do_update_mu:
            obj.update_mu()
            
        if obj.do_update_sigma:
            obj.update_sigma()
#        table(obj.component_names, ...
#            arrayfun(@(x) min(eig(obj.sigma(:,:,x))), 1:size(obj.sigma,3))',...
#            sum(obj.gamma(:,1:end-length(obj.noise_props))>0)')
    
#    def subsample(obj):
#        sample_indices = randsample((1:size(obj.X,1)).T, BayesianGMM.sample, true, obj.sample_prob)
#        obj.X_sample = obj.X(sample_indices,:);
        
        
    def update_gamma_sinkhorn(obj):
        
        N_t = obj.train_indices.sum()
        T, _ = sinkhorn_knopp(np.ones((N_t))/N_t, obj.pi, -obj.lp[obj.train_indices,:], BayesianGMM.lambda_sinkhorn, log=True)
        
        obj.gamma[obj.train_indices,:] = T/np.nansum(T,1)[:,np.newaxis]
        
#        N_t = (1-obj.train_indices).sum()
#        T, _ = sinkhorn_knopp(np.ones((N_t))/N_t, obj.pi, -obj.lp[~obj.train_indices,:], BayesianGMM.lambda_sinkhorn, log=True)
#        
#        obj.gamma[~obj.train_indices,:] = T/np.nansum(T,1)[:,np.newaxis]
        
            
        gamma = obj.gamma[:,np.newaxis,:]
        obj.gamma_weight =  np.tile(obj.X_sample[:,:,np.newaxis],(1,1,obj.mu.shape[0]))*gamma[:,:,:obj.K]
        
        
        
    def update_gamma(obj):
            # Update gamma
        obj.PE = 0;
        
        for k in range(obj.K):
            difs = obj.X_sample - obj.mu[k,:]
            

            A = scipy.linalg.sqrtm(np.linalg.inv(obj.sigma[:,:,k].squeeze()))
            
            prior_ll = - 0.5*(obj.mu[k,:]-obj.prior['mu'][k,:])@np.linalg.inv(obj.prior['sigma'][:,:,k])@(obj.mu[k,:]-obj.prior['mu'][k,:]).T \
                - ((obj.D)/2)*np.log(2*np.pi) - 0.5*np.log(np.linalg.det(obj.prior['sigma'][:,:,k]))
            
            obj.ll[:,k] = -0.5*np.nansum((A@difs.T)**2,0) -0.5*np.log(np.linalg.det(obj.sigma[:,:,k]).squeeze()) -((obj.D)/2)*np.log(2*np.pi)
            obj.lp[:,k] = np.log(obj.pi[k])+obj.ll[:,k]+prior_ll
            obj.LE[:,k] = np.log(obj.pi[k])+obj.ll[:,k]
            obj.PE = obj.PE + prior_ll;
        
        for k in range(obj.K,obj.K+obj.B):
            difs = obj.X_sample[:,3:] - obj.noise_colors[k-obj.K,:]
            
            covar = 20*np.eye(obj.C)
            A = scipy.linalg.sqrtm(np.linalg.inv(covar))
            
            prior_ll = -np.log(obj.u_range).sum()
            obj.ll[:,k] = -1/2*np.nansum((A@difs.T)**2,0) -0.5*np.log(np.linalg.det(covar)) -(obj.C/2)*np.log(2*np.pi)
            obj.lp[:,k] = np.log(obj.pi[k])+obj.ll[:,k]+prior_ll
            obj.LE[:,k] = np.log(obj.pi[k])+obj.ll[:,k]
            obj.PE = obj.PE + prior_ll


        ln = logsumexp(obj.lp,1)
        obj.gamma = np.exp(obj.lp - ln[:,np.newaxis])
        
        gamma = obj.gamma[:,np.newaxis,:]
        obj.gamma_weight =  np.tile(obj.X_sample[:,:,np.newaxis],(1,1,obj.mu.shape[0]))*gamma[:,:,:obj.K]
        
        
        
    def update_mu(obj):
#            % Update obj.mu using obj.multivariate prior
        n_gamma_nu = np.nansum(obj.gamma_weight[obj.train_indices,:,:],0).squeeze().T
        n_gamma = np.nansum(obj.gamma[obj.train_indices,:],0)
        mu_tmp = obj.mu.copy()
        for k in range(obj.K):
            covz = obj.sigma[:,:,k]@np.linalg.inv(obj.prior['sigma'][:,:,k])
            A = n_gamma[k]*np.eye(obj.D) + covz
            B = n_gamma_nu[k,:].T + covz@obj.prior['mu'][k,:].T
            mu_tmp[k,:] = np.linalg.inv(A)@B
        
#        obj.mu[:,:3] = mu_tmp[:,:3].copy()
        obj.mu = mu_tmp
        
        
    def update_sigma(obj):
#            % Update obj.sigma
        for k in range(obj.K):
            gamma = obj.gamma[obj.train_indices,k].copy()
            gamma[gamma<BayesianGMM.eps] = BayesianGMM.eps
            diff = np.sqrt(gamma/np.nansum(gamma,0))[:,np.newaxis]*(obj.X_sample[obj.train_indices,:]-obj.mu[k,np.newaxis])
            diff[np.isnan(diff)] = 0
            sigma_k = diff.T@diff;
            
            
            sigma_k[:3,3:]=0
            sigma_k[3:,:3]=0
                            
#                if any(isnan(sigma_k(:))) || any(isinf(sigma_k(:))):
#                    warning('sigma_k has nan values')
            
            eig_val, eig_vec = np.linalg.eig(sigma_k)
            
#            diag_eig_val = np.array(np.diag(eig_val))
            
            if eig_val.max() > BayesianGMM.eig_bound[1]:
                eig_val[eig_val>BayesianGMM.eig_bound[1]] = BayesianGMM.eig_bound[1]
            
            if eig_val.min() < BayesianGMM.eig_bound[0]:
                eig_val[eig_val<BayesianGMM.eig_bound[0]] = BayesianGMM.eig_bound[0]
            
            eig_val = np.diag(eig_val)
            
            obj.sigma[:,:,k] = (eig_vec@eig_val@eig_vec.T).copy()
        
        
    def update_pi(obj):
#            % Update obj.pi
        obj.pi = np.nansum(obj.gamma[obj.train_indices,:],0)
        obj.pi = obj.pi/obj.pi.sum()
            
    
    def log_likelihood(obj):
        le = np.array([np.nansum(np.log(np.nansum(np.exp(obj.LE[ obj.train_indices,:]),1)))+obj.PE, \
                       np.nansum(np.log(np.nansum(np.exp(obj.LE[~obj.train_indices,:]),1)))+obj.PE])
        obj.le = np.vstack((obj.le,le[np.newaxis,:]))
        

    def check_convergence(obj):
        obj.converged = len(obj.le)>1 and np.abs(obj.le[-2]-obj.le[-1]) < BayesianGMM.converge
        
    
    def update_prior(obj):
#        tr,beta0,beta = AutoID.auto_id([],obj.prior,obj.mu,n_iter=1,alpha=0,degree=1)
#        obj.prior['mu'][:,:3] = tr[:,:3]
        
        
        beta_pos = Helpers.MCR_solver(obj.prior['mu'][:,:3], np.concatenate((obj.mu[:,:3],np.ones((obj.K,1))),1), obj.prior['sigma'][:3,:3,:])
        beta_col = Helpers.MCR_solver(obj.prior['mu'][:,3:], np.concatenate((obj.mu[:,3:],np.ones((obj.K,1))),1), obj.prior['sigma'][3:,3:,:])
        
        
        beta_pos = np.concatenate((beta_pos, np.array([[0,0,0,1]]).T),1)
        beta_col = np.concatenate((beta_col, np.append(np.zeros((obj.C)),1)[:,np.newaxis]), 1)
        
#        beta_pos = np.eye(4)
#        beta_col = np.eye(obj.C+1)
#        
        inv_beta_pos = np.linalg.inv(beta_pos)
        inv_beta_col = np.linalg.inv(beta_col)
        
        beta = np.zeros((4+obj.C,3+obj.C))
        beta[[0,1,2,3+obj.C],:3] = inv_beta_pos[:,:3]
        beta[np.append(np.arange(3,3+obj.C),3+obj.C),3:3+obj.C] = inv_beta_col[:,:obj.C]
        

        obj.prior['mu'] = np.concatenate((obj.prior['mu'],np.ones((obj.prior['mu'].shape[0],1))),1)@beta
        
        for neuron in range(obj.K):
#            obj.prior['sigma'][:3,:3,neuron] = beta[:,:3]@obj.prior['sigma'][:3,:3,neuron]@beta[:,:3].T
            obj.prior['sigma'][:,:,neuron] = beta[:-1,:].T@obj.prior['sigma'][:,:,neuron]@beta[:-1,:]
            obj.prior['sigma'][:,:,neuron] = (obj.prior['sigma'][:,:,neuron]+obj.prior['sigma'][:,:,neuron].T)/2
        
    def annotate(obj, neurons, mu):
        atlas_ind = np.array([obj.prior['names'].index(x) for x in neurons])
        obj.prior['mu'][atlas_ind,:] = mu
        obj.prior['sigma'][:,:,atlas_ind] = 10*obj.prior['sigma'][:,:,atlas_ind]/obj.prior['sigma'].shape[2]
        obj.mu[atlas_ind,:] = obj.prior['mu'][atlas_ind,:].copy()
    
    def compute_accuracy(obj, annotations, positions, scale, radius=3):
        
        image_ind = np.array([annotations.index(x) for x in obj.prior['names'] if x in annotations])
        atlas_ind = np.array([obj.prior['names'].index(x) for x in obj.prior['names'] if x in annotations])
        
        distances = np.sqrt((((obj.mu[atlas_ind,:3] - positions[image_ind,:])*scale)**2).sum(1))
        within_radius = distances < radius
        acc = within_radius.sum()/len(within_radius)
        mse = distances.mean()
        return acc, mse
    
    def compute_ranking_accuracy(obj, annotations, positions, p_mtx, scale=1, radius=3, top_k=1):
        order = (-p_mtx).argsort(0)
        rankings = order.argsort(0)
        ranks = []
        for image_ind, neuron in enumerate(annotations):
            if neuron not in obj.prior['names']:
                continue
            
            atlas_ind = obj.prior['names'].index(neuron)
            
            distances = np.sqrt((((obj.mu[:,:3] - positions[image_ind,:])*scale)**2).sum(1))
            
            
            if distances.min() < radius:
                ranks.append(rankings[atlas_ind,distances < radius].min())
            else:
                ranks.append(np.nan)
        
        print(np.array(ranks))
        return np.where(np.array(ranks) < top_k)[0].shape[0]/len(ranks)
    
    def __init__(obj,atlas,data,noise_props=np.array([.25,.5,.2]),random_init=False):
        obj.converged = -1
        obj.le = np.array([[],[]]).T
        obj.timing = np.array([])
        
        obj.noise_props  = noise_props
        obj.noise_colors = np.array([[0,1,3],\
                                    [0,4,0],\
                                    [0,0,2]])
                    
        obj.do_sinkhorn = True
        obj.do_update_mu = True
        obj.do_update_pi = False
        obj.do_update_sigma = False
        obj.do_subsample = False
        
        obj.X = data
        
        obj.prior = copy.deepcopy(atlas)
        obj.prior['sigma'] = obj.prior['sigma']/10
        
        if random_init:
            obj.mu = data[np.random.randint(data.shape[0], size=obj.prior['mu'].shape[0]), :]
        else:
            obj.mu = obj.prior['mu'].copy()
        
        obj.D = obj.prior['mu'].shape[1]
        obj.K = obj.prior['mu'].shape[0]
        obj.C = obj.D-3
        obj.N = obj.X.shape[0]
        obj.B = len(obj.noise_props)
        
        obj.pi = np.hstack(((1-obj.noise_props.sum(0))*np.ones((obj.K))/obj.K, obj.noise_props))
        obj.sigma = 10*np.tile(np.eye(obj.D)[:,:,np.newaxis],(1,1,obj.K))
        
        obj.u_range = obj.X.max(0)[:3]-obj.X.min(0)[:3]
        obj.X_sample = obj.X
        obj.train_indices = np.random.binomial(1,0.8,obj.N).astype(bool)
    
        obj.ll = np.zeros((obj.N,obj.K+obj.B))
        obj.lp = np.zeros((obj.N,obj.K+obj.B))
        obj.LE = np.zeros((obj.N,obj.K+obj.B))
        obj.PE = np.array(0)
        

    def visualize(obj,data,titlestr,scale=1):
        sz = np.array([np.linalg.eig(obj.sigma[:3,:3,i])[0].sum() for i in range(obj.sigma.shape[2])])
        cl = obj.mu[:,3:6]; cl[cl < 0] = 0; cl = cl/cl.max()
        P = obj.mu[:,:3]*scale
        plt.cla()
        plt.imshow(10*data[:,:,:,[0,1,2]].max(2))
        plt.scatter(P[:,1],P[:,0],s=10*sz,edgecolors='w',marker='o',facecolors=cl)
        
        cl = obj.prior['mu'][:,3:]; cl[cl < 0] = 0; cl = cl/cl.max()
        P = obj.prior['mu'][:,:3]*scale
        plt.scatter(P[:,1],P[:,0],s=10*sz,edgecolors='w',marker='x',facecolors=cl)
        
        plt.show()
        plt.title(titlestr)
        plt.axis('off')

        
        for i in range(len(obj.prior['names'])):
            plt.annotate(obj.prior['names'][i],(P[i,1],P[i,0]),c='r')
        
        plt.pause(.1)
        
    def visualize_segmentation(obj,scale=1,colored=False,titlestr=''):
        a = Utils.obs2image(obj.X[:,:3], obj.gamma, scale)
#        b = a[:,:,:,:obj.K]@obj.mu[np.newaxis,np.newaxis,:,3:]
#        b_n = a[:,:,:,obj.K:]@obj.noise_colors[np.newaxis,np.newaxis,:,:]
#        plt.subplot(212)
#        plt.imshow(b.max(2)/b.max())
#        plt.title('Segmentation')
#        plt.axis('off')
#        plt.grid('on')
        
#        plt.subplot(313)
#        plt.imshow(5*b_n.max(2)/b_n.max())
#        plt.title('Noise Components')
#        plt.axis('off')
#        plt.grid('on')
        
#        plt.subplot(211)
#        a = Utils.obs2image(obj.X[:,:3], obj.X[:,3:], scale)
#        plt.imshow(a.max(2)/a.max())
#        plt.title('Data')
#        plt.axis('off')
#        plt.grid('on')
        
        
        
        
        for k in range(obj.K):
            plt.subplot(1,obj.K+1,k+2)
            im  = plt.imshow(a[:,:,:,k].max(2),cmap='Greys',vmin=0,vmax=1)
            plt.gca().axes.get_xaxis().set_ticks([])
            plt.gca().axes.get_yaxis().set_ticks([])
            plt.scatter(obj.mu[k,1],obj.mu[k,0],s=20,c='r')
        
        plt.subplot(1,obj.K+1,1)
        a = Utils.obs2image(obj.X[:,:3], obj.X[:,3:], scale)
        plt.imshow(a.max(2)/a.max())
        plt.title('Data')
        plt.gca().axes.get_xaxis().set_ticks([])
        plt.gca().axes.get_yaxis().set_ticks([])
            
        cb_ax = plt.gcf().add_axes([0.92, 0.3, 0.01, 0.4])
        plt.colorbar(im, cax=cb_ax)
        
        plt.show()
        plt.title(titlestr)

