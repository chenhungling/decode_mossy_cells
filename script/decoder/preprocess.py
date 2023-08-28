# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 12:29:20 2021

@author: Hung-Ling
"""
import numpy as np

# %%
def bin_data(F, axis=0, window=4, statistic='mean'):
    '''
    Bin sequence of data into defined time bin by summing or averaging every {window} points.

    Parameters
    ----------
    F : np.ndarray (1D or 2D)
        Time series data set.
        Shape (# time points, # features) if axis=0
        Shape (# features, # time points) if axis=1  
    axis : int 0|1
        The axis along which to bin data
    window : int
        The number of consecutive data points to be merged. The default is 4
    statistic : string 'sum'|'mean'|'median'
        The method used to compute a statistic in the window. The default is 'mean'

    Returns
    -------
    F2 : np.ndarray (2D)
        Binned data set.
        Shape (# binned time points, # features) if axis=0
        Shape (# features, # binned time points) if axis=1      
        Note (# binned time points) = (# times points)//window - remainders at the end are dropped
    '''
    if F.ndim == 1:  # Make F into 2D array
        if axis == 0:
            F = F.reshape((-1,1))
        else:
            F = F.reshape((1,-1))
    if axis == 0:
        ntime, nfeature = F.shape
        nbin = int(ntime/window)
        if statistic == 'sum':
            F2 = np.sum(F[:nbin*window, :].reshape((window, nbin, nfeature), order='F'), axis=0)
        elif statistic == 'mean':
            F2 = np.mean(F[:nbin*window, :].reshape((window, nbin, nfeature), order='F'), axis=0)
        elif statistic == 'median':
            F2 = np.median(F[:nbin*window, :].reshape((window, nbin, nfeature), order='F'), axis=0)
    else:
        nfeature, ntime = F.shape
        nbin = int(ntime/window)
        if statistic == 'sum':
            F2 = np.sum(F[:, :nbin*window].reshape((nfeature, nbin, window)), axis=2)  # Default order='C'
        elif statistic == 'mean':
            F2 = np.mean(F[:, :nbin*window].reshape((nfeature, nbin, window)), axis=2)
        elif statistic == 'median':
            F2 = np.median(F[:, :nbin*window].reshape((nfeature, nbin, window)), axis=2)
    return F2

# %%
def convolve_data(G, axis=0, bins_before=2, bins_after=2):
    '''
    Combine data across a specified sliding window (i.e. convolved with constant ones).
    This apply to neural data to get the total number of spikes across "bins_before", "bins_current=1", and "bins_after" 

    Parameters
    ----------
    G : np.ndarray (1D or 2D)
        (Binned) time series data set. Shape (# time points, # cells)
    axis : int 0|1
        The axis along which to convolve data
    bins_before : int
        Nmber of bins preceding the concurrent bin. The default is 2
    bins_after : int
        Number of bins following the concurrent bin. The default is 2

    Returns
    -------
    G2 : Data set combined across defined time bins.
    Note the length of G2 is shorter than that of G by (bins_before + bins_after)
    '''
    if G.ndim == 1:  # Make G into 2D array
        if axis == 0:
            G = G.reshpae((-1,1))
        else:
            G = G.reshpae((1,-1))
    b = bins_before+bins_after+1  # +1 for the concurrent bin
    if axis == 0:
        nbin, ncell = G.shape
        G2 = np.empty((nbin-b+1, ncell))
        for i in range(ncell):
            G2[:, i] = np.convolve(G[:, i], np.ones(b), mode='valid')  # Length nbin-b+1
    else:
        ncell, nbin = G.shape
        G2 = np.empty((ncell, nbin-b+1))
        for i in range(ncell):
            G2[i] = np.convolve(G[i], np.ones(b), mode='valid')  # Length nbin-b+1
    return G2

