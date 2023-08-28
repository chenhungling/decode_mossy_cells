# -*- coding: utf-8 -*-
"""
Take the invivo data file (*.hdf5) (see suite2p_workflow.py),
perform cell-by-cell analysis (activities, spatial information, reliability, ...)

Short notations:
    m: mouse iterable (n_mice: number of animals)
    k: category iterable (ncat: number of categories/sessions)
    c: context iterable (nctx: number of context, 2 for Fam and Nov)
    i: cell iterable (ncell: total number of cells, ncells: number of cells per animal)
Common variables:
    data : dict, value type is list (mouse) of list (category) of arrays
        't': number of time points for each trial, shape (ntrial,)
        'F': fluorescence traces, shape (ncell,T)
        'Tr': transients, shape (ncell,T)
        'y': position, shape (T,)
        'moving': period with speed > 5 cm/s, boolean array, shape (T,)
    cells : dict, value type is list (mouse) of array
        'trate': transient rate, shape (ncell, nctx)
        'tauc': transient auc, shape (ncell, nctx)
        'si_unbiased': spatial information, shape (ncell, nctx)
        'si_pvalue': p-value of spatial information, shape (ncell, nctx)
    tuning : list (category) of arrays, shape (ncell, ybin, ntrial)
        Spatial tuning of each category
        
@author: Hung-Ling
"""
import os
from glob import glob
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir(r'C:\Users\Hung-Ling\Documents\Analysis_Python\MC_Analysis')
plt.style.use('figure.mplstyle')

from function.data2p import get_data_bis
import function.tuning_function as tf
import function.utils as ut
import function.display as disp

# %% Setup
folder = r'D:\LW_Alldata\SI_Transient'
alldata = {'Enriched': glob(os.path.join(folder,'Enriched*.hdf5')),
           'Dissimilar': glob(os.path.join(folder,'Distinct*.hdf5')),
           'Similar': glob(os.path.join(folder,'Similar*.hdf5'))}

datapath = alldata['Enriched'][2]
day = 1  # Recording days, None for all days
min_rate = 1/60  # Minimum transient rate for active cells
pval_thr = 0.01  # Threshold for significant spatial information
ybin = 80  # Number of spatial bins
ysigma = 1  # Size of the Gaussian filter (in number of spatial bin) for smoothing the tuning curves

# %% Load data
data, cells, days, ctx, selected_cells = get_data_bis(
    datapath, day=day, min_rate=min_rate, verbose=True)

with h5py.File(datapath, 'r') as f:
    yrange = f['params/yrange'][()]  # Dataset dependent
    
print('Recording days:', days)
print('Context fam/nov:', ctx)

# %% Place cells
def get_place_cells(cells, pval_thr=0.05):
    
    significant = (cells['si_pvalue'] <= pval_thr).T  # (nctx, ncell) if one day
    place_cells = np.any(significant, axis=0)
    PC = np.array([np.sum(np.all(~significant, axis=0)),  # None
                   np.sum(significant[0] & ~significant[1]),  # Exclusively Fam
                   np.sum(np.all(significant, axis=0)),  # Both
                   np.sum(significant[1] & ~significant[0])])  # Exclusively Nov
    
    return place_cells, PC

place_cells, PC = get_place_cells(cells, pval_thr=pval_thr)
ncell = data['F'][0].shape[0]
print('-'*36)
print('Active cells: ', ncell)
print('Place cells: ', np.sum(place_cells))
print('Proportion of place cells %.2f %%' % (np.sum(place_cells)/ncell*100))
print('PC in both %.2f %%' % (PC[2]/ncell*100))
print('PC exclusively in Fam %.2f %%' % (PC[1]/ncell*100))
print('PC exclusively in Nov %.2f %%' % (PC[3]/ncell*100))
print('-'*36)
    
# %%
ut.compare_paired([cells['trate'].T], varnames=['Mouse','Context','Activity'], 
                  varvalues=[np.arange(1)+1,['Fam','Nov'],None], dodge=True, alpha=0.8)
ut.compare_paired(ut.discard_zero([cells['si_unbiased'].T]), varnames=['Mouse','Context','SI'], 
                  varvalues=[np.arange(1)+1,['Fam','Nov'],None], dodge=True, alpha=0.8)

# %% Spatial tuning
tuning = tf.compute_tuning(data, yrange=yrange, ybin=ybin, ysigma=ysigma,
                           spike=False, transient=True)
## Clipping normalization
# tuning_ = tf.normalize_tuning(tuning, prct=95)

# %% Display data
disp.show_trace(data, spike=False)

disp.show_tuning(tuning)

disp.show_trial(data, tuning, spike=False, ctx=ctx)

disp.draw_trace(data, tuning, spike=False, cell=0, trials=[[2,3,4,6,7,8,9],[],[1,2,3,4,6,7,8,9],[]])

# %% Example tuning curves
n = 16  # Number of cells to show

pc = np.where(place_cells)[0]
rate = np.mean(cells['trate'], axis=1)[pc]
neuron = np.argsort(rate)[-n::]  # Top active n neurons
score = np.zeros(n)
for i, ii in enumerate(neuron):
    # score[i] = cells['remapping'][pc][ii]
    score[i] = cells['discrimination'][pc][ii]
order = np.argsort(score)[::-1]
print('Place cells ordered by score: \n ', pc[neuron][order])

fig, ax = plt.subplots()
ax.stem(range(len(score)), score[order])

disp.draw_tuning(tuning, pc[neuron][order], ctx=ctx, prct=98)
# disp.draw_tuning(tuning, [50,28,0,14,20,69,58,2,25,47], ctx=ctx, prct=98)

# %% Explore scatterplot matrix
def explore_correlation(cells):
    
    ## Pool context
    df1 = pd.DataFrame({'Activity': cells['tauc'].ravel(),
                        'Spatial information': cells['si_unbiased'].ravel(),
                        'Reliability': cells['reliability'].ravel()})
    sns.pairplot(df1)
    
    df2 = pd.DataFrame({'Fmean': cells['tauc'].mean(axis=1),
                       'Fdiff': cells['tauc'][:,1]-cells['tauc'][:,0], # /(cells['tauc'][:,1]+cells['tauc'][:,0]),
                       'Reliability': np.nanmean(cells['reliability'], axis=1),
                       'Remapping': cells['remapping'],
                       'Discrimination': cells['discrimination']})
    sns.pairplot(df2)
    
explore_correlation(cells)

# %% Run for all datasets
datapaths = [alldata['Enriched'][2]]  # Enriched, Dissimilar, Similar
pc_list = []
tuning_list = []

for m, datapath in enumerate(datapaths):
    
    data, cells, days, ctx = get_data_bis(datapath, day=day, min_rate=min_rate)
    
    with h5py.File(datapath, 'r') as f:
        yrange = f['params/yrange'][()]  # Dataset dependent

    place_cells, PC = get_place_cells(cells, pval_thr=pval_thr)
    pc_list.append(place_cells)
    
    tuning = tf.compute_tuning(data, yrange=yrange, ybin=ybin, ysigma=ysigma,
                               spike=False, transient=True)
    ## Clipping normalization
    # tuning_ = tf.normalize_tuning(tuning, prct=95)
    tuning_list.append(tuning)
    
# %% Display tuning maps of the two contexts
def display_tuning(tuning_list, ctx=np.array([0,1,0,1]), sort=0):
    
    nctx = len(set(ctx))
    tuning_mean = [[] for _ in range(nctx)]
    mice = range(len(tuning_list))

    for c in range(nctx):
        for m in mice:
            tuning = tuning_list[m]
            M = np.concatenate([tuning[k] for k in np.where(ctx==c)[0]], axis=2)
            tuning_mean[c].append(M.mean(axis=2))
            
        tuning_mean[c] = np.vstack(tuning_mean[c])
        
    fig, axs = plt.subplots(1,nctx,figsize=(5,4),sharey=True)
    ncell = tuning_mean[0].shape[0]
    order = ut.sort_tuning(tuning_mean[sort])[1]
    for c in range(nctx):
        axs[c].imshow(tuning_mean[c][order], cmap='jet', interpolation='none',
                      extent=[0,4,ncell,0])
    for ax in axs:
        ax.set_aspect('auto')
        ax.set_xlabel('Position (m)')
    axs[0].set_ylabel('Cell')
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.1)
    
display_tuning(tuning_list, ctx=ctx, sort=0)

# %% Display tuning maps of place cells
tuning_pc_list = []
for m in range(len(tuning_list)):
    pc = pc_list[m]
    tuning_pc_list.append([tun[pc] for tun in tuning_list[m]])

display_tuning(tuning_pc_list, ctx=ctx, sort=0)

