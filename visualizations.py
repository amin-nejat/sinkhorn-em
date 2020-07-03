# -*- coding: utf-8 -*-

import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse
from BUtils import Utils
import matplotlib.pyplot as plt
import numpy as np
import imageio


def confidence_ellipse(mu, sigma, ax, n_std=3.0, facecolor='none', **kwargs):
    pearson = sigma[0, 1]/np.sqrt(sigma[0, 0] * sigma[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
 
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    scale_x = np.sqrt(sigma[0, 0]) * n_std
    scale_y = np.sqrt(sigma[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mu[0], mu[1])

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def visualize_average_segmentation(X,µ,Σ,ems,labels,save=True,folder=''):
    K  = µ.shape[0]
    
    for i in range(3):
        a = Utils.obs2image(X.numpy()[:,:3], np.array([em.gamma for em in ems[i]]).mean(0), 1)
        mu = np.array([em.mu for em in ems[i]])
        for k in range(K):
            plt.subplot(3,K+1,i*(K+1)+k+2)
            im  = plt.imshow(a[:,:,:,k].max(2),cmap='Greys',vmin=0,vmax=1)
            confidence_ellipse(µ[k,[1,0]].numpy(), Σ[k,:2,:2].numpy(), plt.gca(), n_std=3.0, edgecolor='g')
            plt.gca().axes.get_xaxis().set_ticks([])
            plt.gca().axes.get_yaxis().set_ticks([])
            plt.scatter(mu[:,k,1],mu[:,k,0],s=20,c='r')
            
            
            
            
        plt.subplot(3,K+1,i*(K+1)+1)
        a = Utils.obs2image(X[:,:3], X[:,3:], 1)
        plt.imshow(a.max(2)/a.max())
        plt.gca().axes.get_xaxis().set_ticks([])
        plt.gca().axes.get_yaxis().set_ticks([])
        plt.ylabel(labels[i])
        
    cb_ax = plt.gcf().add_axes([0.92, 0.1, 0.01, 0.8])
    plt.colorbar(im, cax=cb_ax)
    
    plt.show()
    
    if save:
        plt.savefig(folder+'\\average_segmentation.eps',format='eps')
        plt.savefig(folder+'\\average_segmentation.png',format='png')
        plt.close('all')
    

def visualize_iteration(data,ems,labels,save=True,name='iter',folder=''):
    for i in range(3):
        plt.subplot(1,3,i+1)
        ems[i].visualize(data,labels[i])
        
    if save:
        plt.savefig(folder+'\\'+name)
        plt.close('all')
    
    for i in range(3):
        ems[i].visualize_segmentation()
        if save:
            plt.savefig(folder+'\\'+name+'-seg-'+labels[i])
            plt.close('all')

def visualize_animated_gif(n_iter,n_trials,folder='',labels=None):
    
    for trial in range(n_trials):
        images = []
        for i in range(n_iter):
            try:
                filename = folder+'\\trial-'+str(trial)+' iter-'+str(i)+'.png'
                images.append(imageio.imread(filename))
            except:
                break
        
        imageio.mimsave(folder+'\\trial-'+str(trial)+'.gif', images)
        
    
    if labels is not None:
        for k in range(len(labels)):
            for trial in range(n_trials):
                images = []
                for i in range(n_iter):
                    try:
                        filename = folder+'\\trial-'+str(trial)+' iter-'+str(i)+'-seg-'+labels[k]+'.png'
                        images.append(imageio.imread(filename))
                    except:
                        break
                
                imageio.mimsave(folder+'\\trial-'+str(trial)+'-seg-'+labels[k]+'.gif', images)
    
def visualize_evaluation(ems_all, labels, save=True, folder=''):
    ems_sz = [np.array([oem.le.shape[0] for oem in tem]).min() for tem in ems_all]
    
    ems = [np.array([oem.le[:ems_sz[i],:] for oem in ems_all[i]]) for i in range(len(ems_all))]
    acs = [[em.acc[:ems_sz[i]] for em in ems_all[i]] for i in range(len(ems_all))]
    
    timing = [np.array([oem.timing[:ems_sz[i]] for oem in ems_all[i]]).mean(0).cumsum() for i in range(len(ems_all))]
    
    print(timing)
    
    plt.figure()
    
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times New Roman"
    
    for i in range(len(ems)):
        plt.subplot(221)
        
        plt.plot(timing[i],ems[i].mean(0)[:,0])
        plt.fill_between(timing[i], ems[i].mean(0)[:,0]-ems[i].std(0)[:,0], 
                         ems[i].mean(0)[:,0]+ems[i].std(0)[:,0], alpha=0.2)
        plt.title('Train Log Likelihood',fontsize= 20)
        plt.grid()

        plt.subplot(222)
        plt.plot(timing[i],ems[i].mean(0)[:,1])
        plt.fill_between(timing[i], ems[i].mean(0)[:,1]-ems[i].std(0)[:,1], 
                         ems[i].mean(0)[:,1]+ems[i].std(0)[:,1], alpha=0.2)
        
        plt.title('Test Log Likelihood',fontsize= 20)
        plt.grid()


        plt.subplot(223)
        plt.plot(timing[i],np.array(acs[i]).mean(0)[:,0])
        plt.fill_between(timing[i], np.array(acs[i]).mean(0)[:,0]-np.array(acs[i]).std(0)[:,0], 
                         np.array(acs[i]).mean(0)[:,0]+np.array(acs[i]).std(0)[:,0], alpha=0.2)
        plt.title('Accuracy',fontsize= 20)
        plt.grid()
        plt.xlabel('Time (s)',fontsize=15)

        plt.subplot(224)
        plt.plot(timing[i],np.array(acs[i]).mean(0)[:,1])
        plt.fill_between(timing[i], np.array(acs[i]).mean(0)[:,1]-np.array(acs[i]).std(0)[:,1], 
                         np.array(acs[i]).mean(0)[:,1]+np.array(acs[i]).std(0)[:,1], alpha=0.2)
        plt.title('MSE',fontsize= 20)
        plt.grid()
        plt.xlabel('Time (s)',fontsize=15)
        
    plt.legend(labels)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()

    plt.pause(.1)
    
    if save:
        plt.savefig(folder+'\\evaluation.eps',format='eps')
        plt.savefig(folder+'\\evaluation.png',format='png')
        plt.close('all')        


