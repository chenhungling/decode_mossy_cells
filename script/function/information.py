# -*- coding: utf-8 -*-
"""
Information theory related calculations.
Spatial or mutual information between two random variables.

Neuroscience Ref :
Spatial information :
    Skaggs, W. E., McNaughton, B. L. & Gothard, K. M. An Information-Theoretic Approach to Deciphering the Hippocampal Code.
    in Advances in Neural Information Processing Systems 5, [NIPS Conference] 1030–1037
Mutual information :
    Panzeri, S. et al. Correcting for the Sampling Bias Problem in Spike Train Information Measures.
    Journal of Neurophysiology 98, 1064–1072 (2007)
    Souza, B. C. et al. On Information Metrics for Spatial Coding. Neuroscience 375, 62–73 (2018)

@author: Hung-Ling
"""
import numpy as np

from function.tuning_function import tuning_1d

# %%
def entropy(pmf):
    '''
    Parameters
    ----------
    pmf : np.ndarray (1D)
        Vector of discrete Probability Mass Function.
        Assume pmf is a well-formed distribution, that is,
        np.sum(pmf)==1 and not np.any(pmf<0)

    Returns
    -------
    float
        Shanon entropy of the distribution (PMF) in bits.
    '''
    pmf = pmf[pmf>0]  # Reduce to non-zero entries to avoid log(0)
    h = -np.sum(pmf*np.log2(pmf))  # Shanon entropy (in bits)
    return np.abs(h)  # Return absolute value (avoid -0)

# %%
def spatial_information_1d(activity, ys, yrange=None, ybin=80, fps=20, spike=False):
    '''
    Parameters
    ----------
    activity : np.ndarray (1D or 2D)
        Fluorescence trace or firing rate, shape (ncell, T)
    ys : np.ndarray (1D)
        y positions, shape (T,)
    yrange : list or tuple
        The edge coordinates (ymin, ymax)
    ybin : int
        Number of y bins. Note that the number of bin edges will be ybin+1
    fps : float
        Frame per second
    spike : bool
        False : return bits per second
        True : return bits per spike

    Returns
    -------
    spatial_info : np.ndarray (1D), shape (ncell,)
        Spatial information (bits/sec or bits/spike)
    '''
    if activity.ndim == 1:
        activity = activity[np.newaxis, :]  # Make it 2D array of shape (ncell, T)
        
    ncell = activity.shape[0]
    
    ## Fraction of time (probability) and mean activity in position bin
    occupancy, tuning = tuning_1d(activity, ys, yrange=yrange, ybin=ybin)
    occupancy = occupancy/np.sum(occupancy)  # Normalized to unit sum, may contain zero
    tuning[np.isnan(tuning)] = 0
    
    ## Overall mean activity
    mean_activity = np.mean(activity, axis=1)
    
    spatial_info = np.zeros(ncell)
    for i in range(ncell):
        idx = (tuning[i] > 0)  # Avoid convergence problem of 0*log(0)
        if idx.any():  # Otherwise 0
            info = occupancy[idx]*tuning[i,idx]*np.log2(tuning[i,idx]/mean_activity[i])
            if spike:
                spatial_info[i] = np.sum(info/mean_activity[i])  # [bits/spike]
            else:
                spatial_info[i] = np.sum(info)*fps  # [bits/sec] HLC why fps (?) Values scale with arbitrary multiplicative factor of activity (?) 
            
    return spatial_info

# %%
def mutual_information_1d(active, ys, yrange=None, ybin=80):
    '''
    Parameters
    ----------
    active : np.ndarray (1D or 2D)
        Time series of binarized activity, shape (ncell, T)
    ys : np.ndarray (1D)
        y positions, shape (T,)
    yrange : list or tuple
        The edge coordinates (ymin, ymax)
    ybin : int
        Number of y bins. Note that the number of bin edges will be ybin+1

    Returns
    -------
    MI : np.ndarray (1D), shape (ncell,)
        Mutual information (bits)
    '''
    if yrange is None:
        yrange = (ys.min(), ys.max()+np.finfo(float).eps)
        
    ybin_edges = np.linspace(yrange[0], yrange[1], ybin+1)
    
    if not 'bool' in str(active.dtype):
        active = active.astype(bool)
    if active.ndim == 1:
        active = active[np.newaxis, :]  # Make it 2D row vector
    
    ncell, T = active.shape
    
    H = np.histogram(ys, bins=ybin_edges)[0]  # Shape (ybin,)
    Pstate = H/np.sum(H)  # Occupancy, i.e. probability for the behavior state
    Pactive = np.mean(active, axis=1)  # Probability for active cells
    
    MI = np.zeros(ncell)  # Multual information
    for icell in range(ncell):
        a = active[icell]
        H0 = np.histogram(ys[~a], bins=ybin_edges)[0]
        H1 = np.histogram(ys[a], bins=ybin_edges)[0]
        count0 = np.sum(H0)
        count1 = np.sum(H1)
        if count0>0: 
            H0 = H0/count0
        if count1>0:
            H1 = H1/count1
        Pstate0 = H0  # Probability of the behavior state given cell i is inactive
        Pstate1 = H1  # Probability of the behavior state given cell i is active
        nonzero = np.logical_and(Pstate0>0, Pstate1>0)
        
        mi0 = (1-Pactive[icell])*np.sum(Pstate0[nonzero]*np.log2(Pstate0[nonzero]/Pstate[nonzero]))
        mi1 = Pactive[icell]*np.sum(Pstate1[nonzero]*np.log2(Pstate1[nonzero]/Pstate[nonzero]))
        MI[icell] = mi0 + mi1
        
    return MI


    