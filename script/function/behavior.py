# -*- coding: utf-8 -*-
"""
Process behavior data, mainly linear track position in two-photon recordings.

@author: Hung-Ling
"""
import numpy as np
from scipy.interpolate import pchip_interpolate

# %%
def get_behavior_data(filepath, onset=0, offset=np.inf):
    '''
    Get behavior data from the ephys file of Scanbox recordings.

    Parameters
    ----------
    filepath : str
        Full path to the ephys file.
    onset : int, optional
        The first frame to read. The default is 0.
    offset : int, optional
        The last frame to read (Python index -> read actually onset to offset-1).
        The default is np.inf (read until the end)

    Returns
    -------
    y : numpy 1d array
        Position at each frame (If multiple planes -> one position per plane)
    '''
    ephysdata = np.fromfile(filepath, dtype='float32')
    
    ## Determine the number of recording channels assuming constant index for the 10 first frames
    nchan = 1
    while not all(ephysdata[:nchan*10:nchan]==ephysdata[0]):
        nchan += 1
        
    ephysdata = np.reshape(ephysdata, (-1,nchan))  # frame|x|y|lick|eye
    frames = ephysdata[:,0]  # Frame indices (different planes count different frame indices)
    
    if not np.isfinite(offset):
        offset = int(np.max(frames))+1
    
    y = np.zeros(offset-onset)
    for i, t in enumerate(range(onset,offset)):
        if np.any(frames==t):
            y[i] = np.mean(ephysdata[frames==t,2])
        else:
            y[i] = np.NaN
    
    return y

# %%
def smooth_position(y, span=5):
    '''
    Smooth position by downsampling with median and pchip interpolation.

    Parameters
    ----------
    y : numpy 1d array
        Position at each time point.
    span : int, optional
        Size for the median downsampling. The default is 5.

    Returns
    -------
    ysmooth : numpy 1d array
        Smoothed position at each time point.
    '''
    n = len(y)
    
    t = np.arange(n)
    t_split = np.array_split(t, n//span)
    t2 = np.array([np.mean(ts) for ts in t_split])
    y2 = np.array([np.median(y[ts]) for ts in t_split])  # median instead of mean is more robust to outlier
    ysmooth = pchip_interpolate(t2, y2, t)
    
    return ysmooth

# %%
def moving(y, minspeed=5, fps=10.3255, y2cm=100):
    '''
    Compute the speed and determine the running period.

    Parameters
    ----------
    y : numpy 1d array
        Position at each time point.
    minspeed : float, optional
        Minimum speed in cm/s. The default is 5.
    fps : float, optional
        Frame per second (Hz). The default is 10.3255.
    y2cm : float, optional
        Multiplicative factor converting y to cm. The default is 100.

    Returns
    -------
    v : numpy 1d array
        Running speed in cm/s
    ismoving : numpy 1d array of bool
        Indicate the running period.
    '''
    v = y2cm*np.gradient(y)*fps
    ismoving = (v > minspeed)
    
    mid = len(y)//2  # Look at the second half
    start = np.argmin(y[:mid])  # Find the start point
    ismoving[:start] = False
    peak = mid + np.argmax(y[mid:])  # Find the end point
    ismoving[peak:] = False
    
    return v, ismoving    
    
# %%
# import matplotlib.pyplot as plt

# filepath = r'D:\LW_mc_similar\1_182\200605_182_001.ephys'
# y = get_behavior_data(filepath, onset=1, offset=748)
# ysmooth = smooth_position(y, span=5)
# v, ismoving = moving(ysmooth)
# t = np.arange(len(y))

# fig, axs = plt.subplots(2,1,sharex=True)
# axs[0].plot(t, y, c='tab:blue')
# axs[0].plot(t[ismoving], ysmooth[ismoving], '.', c='tab:red')
# axs[1].plot(t, v, c='tab:green')
# axs[1].set(ylim=[0,40])
# fig.tight_layout()


