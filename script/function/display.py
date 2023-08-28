# -*- coding: utf-8 -*-
"""
Various functions displaying fluorescecne traces and spatial tuning curves.

@author: Hung-Ling
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import function.utils as ut

# %%
def draw_trace(data, tuning, ylength=4, cell=0, trials=None, spike=False):
    '''
    For a given cell, draw fluorescence traces and linear track position of certain trials.
    '''
    ncat = len(data['F'])
    if trials is None:  # All trials
        trials = [np.arange(len(nframes)) for nframes in data['t']]
    ntrial = sum([len(tr) for tr in trials])  # Total number of traces to plot
    spatial = np.zeros((ntrial, tuning[0].shape[1]))
    count = ntrial
             
    fig, axs = plt.subplots(1,2,figsize=(10,6),gridspec_kw={'width_ratios':[3,2]})
    axs[0].set(xlabel='Time (s)', ylabel='Run')
    axs[0].spines[['top','right']].set_visible(False)
    axs[1].set(xlabel='Position (m)', yticklabels=[])
    axs[1].yaxis.set_visible(False)
    axs[1].spines[['top','right','left']].set_visible(False)
    fig.tight_layout()
    fig.subplots_adjust(right=0.85, wspace=0.05)
    cax = plt.axes([0.88, 0.15, 0.02, 0.7])
    
    for k in range(ncat):
        nframes = data['t'][k]
        nframes_ = np.hstack([0, np.cumsum(nframes)])
        for r in trials[k]:
            seg = slice(nframes_[r], nframes_[r+1])
            fluor = data['F'][k][cell, seg]
            t = np.arange(len(fluor))/data['fps']
            # y = data['y'][mouse][k][seg].copy()
            # y = (y - y.min())/(y.max()-y.min())
            # axs[0].plot(t, y+count, c='C0', lw=1)
            axs[0].plot(t, fluor+count, c='gray', lw=1)
            
            spatial[ntrial-count] = tuning[k][cell,:,r]
            if spike:
                spks = data['Sp'][k][cell, seg]
                a = (spks > 0)
                if np.any(a):
                    _, stemlines, _ = axs[0].stem(
                        t[a], spks[a]+count, bottom=count, markerfmt='none', basefmt='none')
                    stemlines.set(color='r', alpha=0.8, linewidth=1.2)
                spatial[ntrial-count] *= data['fps']  # unit [dF/F] per sec
            else:
                active = data['Tr'][k][cell, seg]            
                transient = fluor.copy()
                transient[~active] = np.NaN
                axs[0].plot(t, transient+count, c='r', lw=1)
            count -= 1
            
    axs[0].set(yticks=range(1,ntrial+1), yticklabels=range(ntrial,0,-1))
    img = axs[1].imshow(spatial, cmap='jet', extent=(0,ylength,ntrial-0.5,0.5))
    axs[1].set_ylim([ntrial-0.5,-0.5])
    axs[1].set_aspect('auto')
    fig.colorbar(img, cax=cax)

def draw_tuning(tuning, cells_id, ctx=np.array([0,1,0,1]), prct=99):
    '''
    For given cell's IDs, draw tuning curves of all trials.
    '''
    nctx = len(set(ctx))
    ncat = len(tuning)
    x = [[] for _ in range(ncat)]
    for k in range(ncat):
        x[k] = tuning[k][cells_id].ravel()  # (ncell, ybin, ntrial)
    if prct == 100:
        vmax = np.max(np.hstack(x))
    else:
        vmax = np.percentile(np.hstack(x), prct)
    
    fig, axs = plt.subplots(nctx, len(cells_id), figsize=(10,2.5))
    fig.subplots_adjust(left=0.05, right=0.95, wspace=0.06, hspace=0.04)
    
    for c in range(nctx):
        ks = np.where(ctx==c)[0]
        for i, cell in enumerate(cells_id):
            M = np.vstack([tuning[k][cell].T for k in ks])  # (ybin, ntrial) transposed
            axs[c,i].imshow(M, cmap='hot', vmin=0, vmax=vmax, interpolation='none')
    for ax in axs.ravel():
        ax.axis('off')
        ax.set_aspect('auto')

# %%
def show_trace(data, spike=False):
    '''
    For each cell, display fluorescence traces of all trials.

    Parameters
    ----------
    data : dict
        Simple data structure (see data2p.get_data_bis).
    '''
    fps = data['fps']
    Fs = data['F']
    As = data['Tr']
    if spike:
        Ss = data['Sp']
    ncat = len(Fs)
    ncell = Fs[0].shape[0]
    Tmax = max([max(t) for t in data['t']])
    
    plt.ion()
    fig, axs = plt.subplots(ncat,1,sharex=True,figsize=(10,10))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.1, hspace=0.001)
    ax_slider = plt.axes([0.1, 0.02, 0.8, 0.02])
    slider_cell = Slider(ax_slider, 'Cell', 0, ncell-1, valinit=0)
    
    def update(val):
        icell = int(np.round(slider_cell.val))
        for k in range(ncat):
            axs[k].cla()
            nframes = data['t'][k]
            nframes_ = np.hstack([0, np.cumsum(nframes)])
            ntrial = len(nframes)
            ## Fluorescence intensity (normalized by the max)
            for r in range(ntrial):
                seg = slice(nframes_[r], nframes_[r+1])
                trace = Fs[k][icell, seg]
                if np.max(trace) > 0:
                    trace /= np.max(trace)
                t = np.arange(len(trace))/fps
                ## Display plot from top to bottom: vertical shift (ntrial-1-r)
                axs[k].plot(t, (ntrial-1-r)+trace, color='gray', alpha=0.8, linewidth=1)
                if not spike:
                    active = As[k][icell, seg]
                    transient = trace.copy()
                    transient[~active] = np.NaN
                    axs[k].plot(t, (ntrial-1-r)+transient, color='r', linewidth=1)
                else:
                    spks = Ss[k][icell, seg]
                    a = (spks > 0)
                    if np.any(a):
                        spks /= np.max(spks)
                        _, stemlines, _ = axs[k].stem(
                            t[a], (ntrial-1-r)+spks[a], bottom=(ntrial-1-r),
                            markerfmt='none', basefmt='none')  # linefmt='r'
                        stemlines.set(color='r', alpha=0.8, linewidth=1.5)
                
            axs[k].set(xlim=[0, Tmax/fps], ylim=[-0.5, ntrial+0.5],
                       ylabel='Trial', yticks=np.arange(ntrial),
                       yticklabels=np.arange(ntrial)[::-1])  # Flip: trial from top to bottom
        axs[ncat-1].set_xlabel('Time (s)')
        
    slider_cell.on_changed(update)
    slider_cell.set_val(0)
    def arrow_key_control(event):
        if event.key == 'left':
            new_val = int(slider_cell.val - 1)
            if new_val < 0:
                new_val = 0
            slider_cell.set_val(new_val)

        elif event.key == 'right':
            new_val = int(slider_cell.val + 1)
            if new_val > ncell-1:
                new_val = ncell-1
            slider_cell.set_val(new_val)
        
    fig.canvas.mpl_connect('key_release_event', arrow_key_control)

# %% 
def show_tuning(tuning):
    '''
    For each cell, display tuning curves of all trials.
    
    Parameters
    ----------
    tuning : list (category) of arrays, shape (ncell,ybin,ntrial)
        Spatial tuning for each category
    '''
    ncat = len(tuning)
    ncell = tuning[0].shape[0]
    
    plt.ion()
    fig, axs = plt.subplots(ncat,2,sharex=True,figsize=(8,9))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.1)
    ax_slider = plt.axes([0.1, 0.02, 0.8, 0.02])
    slider_cell = Slider(ax_slider, 'Cell', 0, ncell-1, valinit=0)
    def update(val):
        icell = int(np.round(slider_cell.val))
        for k in range(ncat):
            axs[k,0].imshow(tuning[k][icell].T)
            axs[k,0].set_aspect('auto')
            axs[k,1].cla()
            axs[k,1].plot(tuning[k][icell].mean(axis=1))
            
    slider_cell.on_changed(update)
    slider_cell.set_val(0)    
    def arrow_key_control(event):
        if event.key == 'left':
            new_val = int(slider_cell.val - 1)
            if new_val < 0:
                new_val = 0
            slider_cell.set_val(new_val)
        elif event.key == 'right':
            new_val = int(slider_cell.val + 1)
            if new_val > ncell-1:
                new_val = ncell-1
            slider_cell.set_val(new_val)
        
    fig.canvas.mpl_connect('key_release_event', arrow_key_control)

# %% 
def show_trial(data, tuning, spike=False, ctx=np.array([0,1,0,1])):
    '''
    For each trial, display fluorescence traces and tuning maps of all cells.
    
    Parameters
    ----------
    data : dict
        Simple data structure (see mc_analysis.py/get_mat_data.py).
    tuning : list (category) of arrays, shape (ncell,ybin,ntrial)
        Spatial tuning for each category
    '''
    nctx = len(set(ctx))
    fps = data['fps']
    Fs = data['F']
    As = data['Tr']
    if spike:
        Ss = data['Sp']
    ncell = Fs[0].shape[0]
    ntrials = [len(t) for t in data['t']]
    ntrial = sum(ntrials)
    categories = np.hstack([np.repeat(k, nt) for k, nt in enumerate(ntrials)])
    itrials = np.hstack([np.arange(t) for t in ntrials])
    
    tuning_mean = [[] for _ in range(nctx)]
    for c in range(nctx):
        M = np.concatenate([tuning[k] for k in np.where(ctx==c)[0]], axis=2)    
        tuning_mean[c] = M.mean(axis=2)  # (ncell, ybin)
    order = ut.sort_tuning(tuning_mean[0])[1]
    
    fig, axs = plt.subplots(1,nctx,sharey=True,figsize=(8,6))
    for c in range(nctx):
        axs[c].imshow(tuning_mean[c][order,:], interpolation='none')
        axs[c].set_aspect('auto')
    fig.tight_layout()
    
    plt.ion()
    fig, axs = plt.subplots(1,2,figsize=(12,10),gridspec_kw={'width_ratios':[3,2]})
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12, wspace=0.02)
    ax_slider = plt.axes([0.1, 0.02, 0.8, 0.02])
    slider_trial = Slider(ax_slider, 'Trial', 0, ntrial-1, valinit=0)
    
    def update(val):
        tr = int(np.round(slider_trial.val))  # Total trial index
        k = categories[tr]  # Category
        r = itrials[tr]  # Trial index within category
        axs[0].cla()
        nframes_ = np.hstack([0, np.cumsum(data['t'][k])])
        seg = slice(nframes_[r],nframes_[r+1])
        ## Display fluorescence traces
        for i, icell in enumerate(order[::-1]):  # Plot from bottom to top corresponds to reverse order of the tuning map 
            trace = Fs[k][icell, seg]
            if np.max(trace) > 0:
                trace /= np.max(trace)
            t = np.arange(len(trace))/fps
            ## Display plot from top to bottom: vertical shift (ncell-1-i)
            axs[0].plot(t, trace+i, color='gray', alpha=0.8, linewidth=1)
            
            if spike:
                spks = Ss[k][icell, seg]
                a = (spks > 0)
                if np.any(a):
                    spks /= np.max(spks)
                    _, stemlines, _ = axs[0].stem(
                        t[a], spks[a]+i, bottom=i, markerfmt='none', basefmt='none')  # linefmt='r'
                    stemlines.set(color='r', alpha=0.8, linewidth=1.5)
            else:
                active = As[k][icell, seg]            
                transient = trace.copy()
                transient[~active] = np.NaN
                axs[0].plot(t, transient+i, color='r', linewidth=1)
                
        axs[0].set(ylim=[-0.5, ncell], xlabel='Time (s)', ylabel='Cell')
        
        ## Color moving periods
        moving = np.hstack([0, data['moving'][k][seg].astype(int)])
        t_start = np.where(np.diff(moving) > 0)[0]
        t_stop = np.where(np.diff(moving) < 0)[0]
        if len(t_start) > len(t_stop):
            t_stop = np.hstack([t_stop, len(moving)-1])
        for t1, t2 in zip(t_start, t_stop):
            axs[0].axvspan(t1/fps, t2/fps, color='g', alpha=0.2)
        ## Display tuning map (sorted)
        axs[1].cla()
        axs[1].imshow(tuning[k][order,:,r], interpolation='none')
        axs[1].set_aspect('auto')
        axs[1].set(xlabel='Spatial bin', yticklabels=[])
        
    slider_trial.on_changed(update)
    slider_trial.set_val(0)    
    def arrow_key_control(event):
        if event.key == 'left':
            new_val = int(slider_trial.val - 1)
            if new_val < 0:
                new_val = 0
            slider_trial.set_val(new_val)
        elif event.key == 'right':
            new_val = int(slider_trial.val + 1)
            if new_val > ntrial-1:
                new_val = ntrial-1
            slider_trial.set_val(new_val)
        
    fig.canvas.mpl_connect('key_release_event', arrow_key_control)         
        
        
    
    