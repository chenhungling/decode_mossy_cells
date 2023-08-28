# -*- coding: utf-8 -*-
"""
Compute pairwise correlations of a neuronal population.

@author: Hung-Ling
"""
import os
from glob import glob
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy import stats

os.chdir(r'C:\Users\Hung-Ling\Documents\Analysis_Python\MC_Analysis')
plt.style.use('figure.mplstyle')

from function.data2p import get_data_bis
import function.tuning_function as tf

# %% Functions computing the population correlation vector (PCorr)
def shuffle_trial(Flist, seed=0):
    '''For each cell, shuffle the trial ID but keep the spatial tuning.
    
    Parameters
    ----------
    Flist : list of arrays of shape (ntrial, ybin, ncell)
        Activity arrays in a list of contexts (and maybe of days).
    
    Returns
    -------
    Fshuffled : list of arrays of shape (ntrial, ybin, ncell)
        Activity arrays with trial order shuffled for each cell.
    '''
    rng = np.random.default_rng(seed)
    Fshuffled = []
    for F in Flist:
        Fsh = F.copy()  # (ntrial, ybin, ncell)
        for i in range(F.shape[2]):
            Fsh[:,:,i] = rng.permutation(F[:,:,i])  # Shuffle by rows (trials)
        Fshuffled.append(Fsh)
    return Fshuffled

def compute_pcorr(activity):
    '''
    Parameters
    ----------
    activity : list of arrays of shape (ncell, ybin, ntrial)
        Activity arrays in a list of contexts (and maybe of days).
        
    Returns
    -------
    paircorr : array of shape (nctx*nday, # cell pairs)
        Pairwise correlations
    '''
    paircorr = []
    for M in activity:  # (ncell, ybin, ntrial)
        ncell = M.shape[2]
        A = np.reshape(M, (-1,ncell)).T  # (ncell, ybin*ntrial)
        pcorr = np.ones((ncell,)*2)  # Pairwise correlation
        for i in range(ncell):
            for j in range(i+1,ncell):
                # pcorr[i,j], _ = stats.pearsonr(A[i], A[j])
                pcorr[i,j], _ = stats.kendalltau(A[i], A[j])
                pcorr[j,i] = pcorr[i,j]
        paircorr.append(pcorr[np.triu_indices(ncell, k=1)])
        
    return np.vstack(paircorr)  # (nctx*nday, # cell pairs)

def compute_pair_corr(datapath, day=1, min_rate=1/60, select='or',
                      ybin=1, ysigma=0, seed=0):
    
    ## Load data
    data, cells, days, ctx, selected_cells = get_data_bis(
        datapath, day=day, min_rate=min_rate, select=select, verbose=False)
    
    with h5py.File(datapath, 'r') as f:
        yrange = f['params/yrange'][()]  # Dataset dependent
    
    ## Spatial tuning (ybin=1 computes activity per trial)
    tuning = tf.compute_tuning(
        data, yrange=yrange, ybin=ybin, ysigma=ysigma,
        spike=False, transient=True)
    
    ## Group tuning (list of categories) by context
    nday = len(set(days))
    nctx = len(set(ctx))
    activity = []  # List of contexts and days
    for d in range(nday):
        for c in range(nctx):
            ks = np.where((days==(d+1)) & (ctx==c))[0]
            M = np.concatenate([tuning[k] for k in ks], axis=2)  # (ncell, ybin, ntrial)
            activity.append(np.transpose(M, [2,1,0]))  # (ntrial, ybin, ncell)

    paircorr = compute_pcorr(activity)
    paircorr2 = compute_pcorr(shuffle_trial(activity, seed=seed))
    
    return paircorr, paircorr2
    
# %% Run for one dataset
folder = r'D:\LW_Alldata\SI_Transient'
alldata = {'Enriched': glob(os.path.join(folder,'Enriched*.hdf5')),
           'Dissimilar': glob(os.path.join(folder,'Distinct*.hdf5')),
           'Similar': glob(os.path.join(folder,'Similar*.hdf5'))}
p = 0
exps = ['Enriched','Dissimilar','Similar']
datapath = alldata[exps[p]][2]

paircorr, paircorr2 = compute_pair_corr(datapath, day=1, min_rate=1/60, select='or',
                                        ybin=1, ysigma=0, seed=0)
print('-'*32)
print('Original means (Fam, Nov): (%.4g, %.4g)' % (np.nanmean(paircorr[0]), np.nanmean(paircorr[1])))
print('Shuffled means (Fam, Nov): (%.4g, %.4g)' % (np.nanmean(paircorr2[0]), np.nanmean(paircorr2[1])))

coactive = paircorr[:, np.all(np.isfinite(paircorr), axis=0)]
coactive2 = paircorr2[:, np.all(np.isfinite(paircorr2), axis=0)]

print('-'*32)
print('Original Fam vs Nov: r=%.4g, p=%.4g' % stats.pearsonr(coactive[0], coactive[1]))
print('Shuffled Fam vs Nov: r=%.4g, p=%.4g' % stats.pearsonr(coactive2[0], coactive2[1]))

# %% Joint plot
vmin = max(np.min(coactive)-0.02,-1)
vmax = min(np.max(coactive)+0.02,1)
bins = np.linspace(vmin,vmax,36)
colors = ['tab:red','darkgray']

fig, axs = plt.subplots(2, 2, figsize=(5,5),
                        gridspec_kw={'height_ratios':[1,5],
                                     'width_ratios':[5,1]})
axs[1,0].scatter(coactive[0], coactive[1], c=colors[0], s=1.6, alpha=0.8, rasterized=True)
axs[1,0].scatter(coactive2[0], coactive2[1], c=colors[1], s=1.6, alpha=0.8, rasterized=True)
axs[1,0].set(xlim=[vmin,vmax], ylim=[vmin,vmax],
             xlabel='Correlation (Fam)', ylabel='Correlation (Nov)')

axs[0,0].axvline(np.mean(coactive[0]), ls='--', c=colors[0])
axs[0,0].axvline(np.mean(coactive2[0]), ls='--', c=colors[1])
axs[0,0].hist(coactive[0], bins=bins, histtype='step', edgecolor=colors[0], lw=1.5)  # Fam, original
axs[0,0].hist(coactive2[0], bins=bins, histtype='step', edgecolor=colors[1], lw=1.5)  # Fam, original
axs[0,0].set(xlim=[vmin,vmax], xticklabels=[], ylabel='Pairs')

axs[1,1].axhline(np.mean(coactive[1]), ls='--', c=colors[0])
axs[1,1].axhline(np.mean(coactive2[1]), ls='--', c=colors[1])
axs[1,1].hist(coactive[1], bins=bins, histtype='step', edgecolor=colors[0], lw=1.5, orientation='horizontal')  # Nov, original
axs[1,1].hist(coactive2[1], bins=bins, histtype='step', edgecolor=colors[1], lw=1.5, orientation='horizontal')  # Nov, original
axs[1,1].set(ylim=[vmin,vmax], yticklabels=[], xlabel='Pairs')
axs[0,1].set_axis_off()
fig.tight_layout()
fig.subplots_adjust(hspace=0.12, wspace=0.12)

# %% Run for all datasets
folder = r'D:\LW_Alldata\SI_Transient'
alldata = {'Enriched': glob(os.path.join(folder,'Enriched*.hdf5')),
           'Dissimilar': glob(os.path.join(folder,'Distinct*.hdf5')),
           'Similar': glob(os.path.join(folder,'Similar*.hdf5'))}
exps = ['Enriched','Dissimilar','Similar']
results = dict(Enriched=[], Dissimilar=[], Similar=[])
results2 = dict(Enriched=[], Dissimilar=[], Similar=[])

for p, exp in enumerate(exps):
    print('-'*8 + exp + '-'*8)
    for m, datapath in enumerate(alldata[exp]):
        paircorr, paircorr2 = compute_pair_corr(
            datapath, day=1, min_rate=1/60, select='or', ybin=1, ysigma=0, seed=0)
        results[exp].append(paircorr)
        results2[exp].append(paircorr2)
        
        print('Original means (Fam, Nov): (%.4g, %.4g)' % (np.nanmean(paircorr[0]), np.nanmean(paircorr[1])))
        print('Shuffled means (Fam, Nov): (%.4g, %.4g)' % (np.nanmean(paircorr2[0]), np.nanmean(paircorr2[1])))
        print('-'*32)
        
np.save('pair_correlations_day1.npy', results, allow_pickle=True)
np.save('pair_correlations_day1_shuffled.npy', results2, allow_pickle=True)

