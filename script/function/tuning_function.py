# -*- coding: utf-8 -*-
'''
Compute the tuning function of each cell with respect to an 1D behavioral/stimulus
variable (e.g. linear track position, speed, head direction) 

Two methods are implemented :
    - 'hist' (histogram) : average activity at each bin of the behavioral variable
    - 'kde' : Gaussian Kernel Density Estimate for the distribution P(active|behavior)
       
@author: Hung-Ling
'''
import numpy as np
from copy import deepcopy
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde
## https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html 
from sklearn.neighbors import KernelDensity
## https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html#sklearn.neighbors.KernelDensity

# %%
def tuning_1d(F, y, yrange=None, ybin=80, method='hist', period=None):
    '''
    Parameters
    ----------
    F : np.ndarray (1D or 2D), shape (ncell, T)
        Firing rate or fluorescence intensity
    y : np.ndarray (1D), shape (T,)
        Time series of position or other 1D variable
    yrange : list or tuple of [ymin, ymax]
    ybin : int
        Number of y bins. Note that the number of bin edges will be ybin+1
    method : 'hist'|'kde'
        'hist' - compute average activity at each position bin
        'kde' - Gaussian Kernel Density Estimate (with default Scott's Rule for
                                                  automatic bandwidth determination)
    period : None or float
        A period for the variable y (e.g. 2*pi in the case of head direction angle)
        This is only relevant in 'kde' method where 1D angle is mapped to 2D unit circle

    Returns
    -------
    occupancy : np.ndarray (1D), shape (ybin,)
        How many times the animal is found in each position bin
    tuning_curves : np.ndarray (2D), shape (ncell, ybin)
        Average activity of each neuron at each position bin
        i.e. estimate of the conditional probability distribution P(active|y)
    '''        
    if y.ndim > 1:
        y = np.squeeze(y)  # Make it 1D vector
    if F.ndim == 1:
        F = F.reshape((1,-1))  # Make it 2D array
    
    ncell, T = F.shape
    if len(y) != T:
        raise IndexError('Position vector y must have the same length as F, consider interpolation.')
        
    if yrange is None:
        yrange = (y.min(), y.max()+np.finfo(float).eps)
    else:  # Discard points outside yrange
        y_in = (y >= yrange[0]) & (y < yrange[1])
        F = F[:, y_in]
        y = y[y_in]
           
    ## Create grid points and initialize arrays
    y_edges = np.linspace(*yrange, ybin+1)
    tuning_curves = np.zeros((ncell, ybin))
    
    if method == 'hist':
        occupancy = np.zeros(ybin, dtype=int)  # Same as np.histogram(y, bins=y_edges, density=False)[0]
        iy = np.searchsorted(y_edges, y, side='right')-1  # Same as np.digitize(y, bins=y_coor)-1
        for t, p in enumerate(iy):
            occupancy[p] += 1
            tuning_curves[:,p] += F[:,t]
            
        tuning_curves[:,occupancy>0] = tuning_curves[:,occupancy>0] / occupancy[occupancy>0]  # Average activity
        tuning_curves[:,occupancy==0] = np.NaN  # NaN if the position bin was not visited
    
    elif method == 'kde':
        occupancy = np.histogram(y, bins=y_edges, density=False)[0]
        y_centers = (y_edges[:-1] + y_edges[1:])/2
        if period is None:
            for i in range(ncell):
                a = (F[i] > 0)  # Boolean array for "active" data points
                if a.sum() > 1:  # Require at least 2 points
                    f = F[i,a]  # Non-zero fluorescence/firing rate as weights
                    kde = gaussian_kde(y[a], weights=f)  # Default bw_method='scott'
                    tuning_curves[i] = kde(y_centers)
                else:
                    tuning_curves[i] = 0
        else:
            ds = np.exp(1j*2*np.pi*y/period)
            d_centers = np.exp(1j*2*np.pi*y_centers/period)
            for i in range(ncell):
                a = (F[i] > 0)
                if a.sum() > 1:
                    f = F[i,a]
                    kde = gaussian_kde(np.vstack([ds[a].real, ds[a].imag]), weights=f)
                    tuning_curves[i] = kde(np.vstack([d_centers.real, d_centers.imag]))
                else:
                    tuning_curves[i] = 0
    else:
        raise ValueError(f'{method} is not implemented, use "hist" or "kde"')
        
    return occupancy, tuning_curves

# %%
def tuning_kde_1d(F, y, yrange=None, ybin=80, bandwidth=1.0):
    '''
    Parameters
    ----------
    F : np.ndarray (1D or 2D), shape (ncell, T)
        Firing rate or fluorescence intensity
    y : np.ndarray (1D), shape (T,)
        Time series of position or other 1D variable
    yrange : list or tuple of [ymin, ymax]
    ybin : int
        Number of y bins. Note that the number of bin edges will be ybin+1
    bandwidth : float
        The bandwidth of the kernel
    period : None or float
        A period for the variable y (e.g. 2*pi in the case of head direction angle)
        
    Returns
    -------
    occupancy : np.ndarray (1D), shape (ybin,)
        How many times the animal is found in each position bin
    tuning_curves : np.ndarray (2D), shape (ncell, ybin)
        Density estimate of the tuning function
    '''
    if y.ndim > 1:
        y = np.squeeze(y)  # Make it 1D vector
    if F.ndim == 1:
        F = F.reshape((1,-1))  # Make it 2D array
    ncell, T = F.shape
    
    if yrange is None:
        yrange = (y.min(), y.max()+np.finfo(float).eps)
    else:  # Discard points outside yrange
        y_in = (y >= yrange[0]) & (y < yrange[1])
        F = F[:, y_in]
        y = y[y_in]
    
    ## Create grid points and store results (cell-by-cell)
    y_edges = np.linspace(*yrange, ybin+1)
    occupancy = np.histogram(y, bins=y_edges, density=False)[0]
    tuning_curves = np.zeros((ncell, ybin))
    y_centers = (y_edges[:-1] + y_edges[1:])/2
    
    for i in range(ncell):
        a = (F[i] > 0)  # Boolean array for "active" data points
        if a.any():
            f = F[i,a]  # Non-zero fluorescence/firing rate as weights
            kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)  
            kde.fit(y[a].reshape((-1,1)), sample_weight=f)
            tuning_curves[i] = np.exp(kde.score_samples(y_centers[:, np.newaxis]))
        else:
            tuning_curves[i] = 0
            
    return occupancy, tuning_curves

# %% Compute spatial tunings
def compute_tuning(data, yrange=(0.,4.), ybin=80, ysigma=1, spike=False, transient=True):
    
    ncat = len(data['F'])
    tuning = [[] for _ in range(ncat)]
    for k in range(ncat):
        ntrial = len(data['t'][k])
        nframes = np.hstack([0, np.cumsum(data['t'][k])])
        dFoY = np.zeros((data['F'][k].shape[0], ybin, ntrial))
        for r in range(ntrial):
            indices = slice(nframes[r], nframes[r+1])
            ytmp = data['y'][k][indices]
            movetmp = data['moving'][k][indices]
            if spike:
                Ftmp = data['Sp'][k][:,indices]
            elif transient:
                Ftmp = data['Tr'][k][:,indices].astype(float)
            else:
                Ftmp = data['F'][k][:,indices]*data['Tr'][k][:,indices]
            dFoY[:,:,r] = tuning_1d(Ftmp[:,movetmp], ytmp[movetmp],
                                    yrange=yrange, ybin=ybin)[1]
        dFoY[np.isnan(dFoY)] = 0  # NaN due to unoccupied y position
        dFoY[dFoY < 0] = 0  # Discard unusual transient giving rise to negative fluor signal
        if ysigma > 0:
            dFoY = gaussian_filter1d(dFoY, ysigma, axis=1, mode='nearest')
        tuning[k] = dFoY
            
    return tuning

# %%
def normalize_tuning(tuning, prct=95):
    
    ncat = len(tuning)
    ncell = tuning[0].shape[0]
    tuning2 = deepcopy(tuning)  # deepcopy avoid modify the original tunings
    
    for i in range(ncell):
        M_ = np.hstack([M[i,:,:] for M in tuning])  # (ncell,ybin,ntrial)
        vmax = np.percentile(M_, prct)
        if vmax > 0:
            for k in range(ncat):
                tuning2[k][i] = np.clip(tuning2[k][i]/vmax,0.0,1.0)
    
    return tuning2
    
    