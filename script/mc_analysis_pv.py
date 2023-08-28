# -*- coding: utf-8 -*-
"""
Take the invivo data file (*.hdf5) (see suite2p_workflow.py),
calculate population vector correlations.

@author: Hung-Ling
"""
import os
from glob import glob
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
from statannotations.Annotator import Annotator
from scikit_posthocs import posthoc_dunn

os.chdir(r'C:\Users\Hung-Ling\Documents\Analysis_Python\MC_Analysis')
plt.style.use('figure.mplstyle')

from function.data2p import get_data_bis
import function.tuning_function as tf
import function.utils as ut

# %% Setup
folder = r'D:\LW_Alldata\SI_Transient'
alldata = {'Enriched': glob(os.path.join(folder,'Enriched*.hdf5')),
           'Dissimilar': glob(os.path.join(folder,'Distinct*.hdf5')),
           'Similar': glob(os.path.join(folder,'Similar*.hdf5'))}
p = 1
exps = ['Enriched','Dissimilar','Similar']
datapath = alldata[exps[p]][1]
day = 1  # Recording days, None for all days
min_rate = 0/60  # Minimum transient rate for active cells
ybin = 80  # Number of spatial bins
ysigma = 2  # Size of the Gaussian filter (in number of spatial bin) for smoothing the tuning curves

# %% Load data
data, cells, days, ctx, _ = get_data_bis(datapath, day=day, min_rate=min_rate, verbose=True)

with h5py.File(datapath, 'r') as f:
    yrange = f['params/yrange'][()]  # Dataset dependent
    
print('Recording days:', days)
print('Context fam/nov:', ctx)

# %% Spatial tuning
tuning = tf.compute_tuning(data, yrange=yrange, ybin=ybin, ysigma=ysigma,
                           spike=False, transient=True)

nctx = len(set(ctx))
tuning_mean = [np.concatenate([tuning[k] for k in np.where(ctx==c)[0]], axis=2).mean(axis=2)
               for c in range(nctx)]

# %% Display tuning
def display_tuning(tuning_mean, sort=0):
    
    fig, axs = plt.subplots(1,len(tuning_mean),figsize=(6,5),sharey=True)
    ncell = tuning_mean[0].shape[0]
    order = ut.sort_tuning(tuning_mean[sort])[1]
    vmax = np.percentile(np.hstack([M.ravel() for M in tuning_mean]),99.9)
    for c in range(nctx):
        axs[c].imshow(tuning_mean[c][order], cmap='jet', interpolation='none',
                      vmin=0, vmax=vmax, extent=[0,4,ncell,0])
        axs[c].set_aspect('auto')
        axs[c].set_xlabel('Position (m)')
    axs[0].set_ylabel('Cell')
    fig.tight_layout()
    
display_tuning(tuning_mean, sort=0)

# %% Population vectors (PVs): even vs odd
def pearson_similarity(X, Y=None):
    '''
    Parameters
    ----------
    X : numpy.ndarray, shape (n_samples_X, n_features)
    Y : numpy.ndarray, shape (n_samples_Y, n_features)

    Returns
    -------
    R : numpy 2d array, shape (n_samples_X, n_samples_Y)
    '''
    if Y is None:
        R = np.corrcoef(X, rowvar=True)
    else:
        R = np.empty((X.shape[0], Y.shape[0]))
        for i, x in enumerate(X):
            if np.var(x) == 0:
                R[i,:] = np.NaN
            else:
                for j, y in enumerate(Y):
                    if np.var(y) == 0:
                        R[i,j] = np.NaN
                    else:
                        R[i,j] = stats.pearsonr(x, y)[0]
    return R
    
def pvcorr(tuning, ctx=np.array([0,1,0,1]), plot=False):
    '''
    Parameters
    ----------
    tuning : list (category) of arrays, shape (ncell,ybin,ntrial)
        Spatial tuning for each category
    '''
    nctx = len(set(ctx))
    tuning_mean = [[] for _ in range(nctx)]
    tuning_12 = [[] for _ in range(nctx)]  # Average over even/odd trials
    for c in range(nctx):
        ks = np.where(ctx==c)[0]
        M = np.concatenate([tuning[k] for k in ks], axis=2)
        tuning_mean[c] = M.mean(axis=2)
        tuning_12[c] = [np.mean(M[:,:,s::2], axis=2) for s in range(2)]
    
    if plot:
        # PVcorr1 = pearson_similarity(tuning_12[0][1].T, tuning_12[0][0].T)  # Fam odd vs even
        # PVcorr2 = pearson_similarity(tuning_mean[1].T, tuning_mean[0].T)  # Nov vs Fam
        PVcorr1 = cosine_similarity(tuning_mean[0].T, tuning_mean[1].T)  # Fam vs Nov
        PVcorr2 = cosine_similarity(tuning_12[1][1].T, tuning_12[1][0].T)  # Nov odd vs even
        
        rs = np.hstack([PVcorr1.ravel(), PVcorr2.ravel()])
        vmax = np.percentile(rs, 99.5)
        
        fig, axs = plt.subplots(2,1,figsize=(3.4,5.4))
        img0 = axs[0].imshow(PVcorr1, interpolation='none', vmin=0,  # cmap='jet'
                             vmax=vmax, extent=[0,4,4,0])
        img1 = axs[1].imshow(PVcorr2, interpolation='none', vmin=0,  # cmap='jet'
                             vmax=vmax, extent=[0,4,4,0])
        axs[0].set(ylabel='Position Fam (m)', xlabel='Position Nov (m)')
        axs[1].set(ylabel='Position Nov odd (m)', xlabel='Position Nov even (m)')
        for ax in axs.ravel():
            ax.invert_yaxis()
        fig.colorbar(img0, ax=axs[0])    
        fig.colorbar(img1, ax=axs[1])
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.3)

    ny = tuning[0].shape[1]
    pvr = np.zeros((3,ny))  # Fam/Nov/Fam-Nov
    for y in range(ny):  # Hard coded here (only for 2 contexts)
        pvr[0,y] = stats.pearsonr(tuning_12[0][0][:,y], tuning_12[0][1][:,y])[0]  # Fam even/odd
        pvr[1,y] = stats.pearsonr(tuning_12[1][0][:,y], tuning_12[1][1][:,y])[0]  # Nov even/odd
        pvr[2,y] = stats.pearsonr(tuning_mean[0][:,y], tuning_mean[1][:,y])[0]  # Fam/Nov
    
    return pvr

pvr = pvcorr(tuning, plot=True)

# %% Population vectors (PVs): train vs test
def display_pvcorr(tuning, ctx=np.array([0,1,0,1]), test=0):
    
    nctx = len(set(ctx))
    tuning_train = [[] for _ in range(nctx)]
    for c in range(nctx):
        ks = np.where(ctx==c)[0]
        M = np.concatenate([tuning[k] for k in ks], axis=2)
        if c == 1:  # Nov
            ntrial = M.shape[2]
            tuning_test = M[:,:,test]
            M = M[:,:,np.setdiff1d(np.arange(ntrial),test)]
        tuning_train[c] = M.mean(axis=2)
    
    PVcorr1 = cosine_similarity(tuning_train[0].T, tuning_test.T)  # Fam vs Nov test
    PVcorr2 = cosine_similarity(tuning_train[1].T, tuning_test.T)  # Nov vs Nov test
    rs = np.hstack([PVcorr1.ravel(), PVcorr2.ravel()])
    vmax = np.percentile(rs, 99)
        
    fig, axs = plt.subplots(2,1,figsize=(4,6))
    img0 = axs[0].imshow(PVcorr1, interpolation='none', vmin=0,  # cmap='jet'
                         vmax=vmax, extent=[0,4,4,0])
    img1 = axs[1].imshow(PVcorr2, interpolation='none', vmin=0,  # cmap='jet'
                         vmax=vmax, extent=[0,4,4,0])
    axs[0].set(ylabel='Position Nov test (m)', xlabel='Position Fam (m)')
    axs[1].set(ylabel='Position Nov test (m)', xlabel='Position Nov (m)')
    for ax in axs.ravel():
        ax.invert_yaxis()
    fig.colorbar(img0, ax=axs[0])    
    fig.colorbar(img1, ax=axs[1])
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3)
    
display_pvcorr(tuning, ctx=ctx, test=7)

# %%
stat, pval = stats.kruskal(*pvr)
print('Kruskal-Wallis test, p-value: %.4g' % pval)
pval_paired = posthoc_dunn(pvr, p_adjust='bonferroni').to_numpy()
print('Post-hoc Dunn\'s test')
print('FF vs NN p-value: %.4g' % pval_paired[0,1])
print('FF vs FN p-value: %.4g' % pval_paired[0,2])
print('NN vs FN p-value: %.4g' % pval_paired[1,2])
print('-'*32)

df = pd.DataFrame(pvr.T, columns=['FF','NN','FN'])

fig, ax = plt.subplots(figsize=(3.5,5))
sns.boxplot(data=df, ax=ax)
ax.set_xticklabels(['FF','NN','FN'])
ax.set_ylabel('PV correlation')

pairs = [('FF','NN'),('FF','FN'),('NN','FN')]
pvalues = [pval_paired[0,1], pval_paired[0,2], pval_paired[1,2]]
pformat = {'pvalue_thresholds': [[1e-3,'***'],[1e-2,'**'],[0.05,'*'],[1,'ns']], 'fontsize': 16}
annot = Annotator(ax, pairs, data=df)
annot.configure(test=None, loc='outside', line_width=1.2, line_height=0.01, pvalue_format=pformat)
annot.set_pvalues(pvalues=pvalues)
annot.annotate()
fig.tight_layout()

# %% Run for all datasets
exps = ['Enriched','Dissimilar','Similar']
pvr_list = []  # Fam/Nov/Fam-Nov

for p, exp in enumerate(exps):
    print('Get %s datasets ...' % exp)
    pvr_pool = []
    for datapath in alldata[exp]:
        data, cells, days, ctx = get_data_bis(datapath, day=day, min_rate=min_rate)
        with h5py.File(datapath, 'r') as f:
            yrange = f['params/yrange'][()]  # Dataset dependent
        tuning = tf.compute_tuning(data, yrange=yrange, ybin=ybin, ysigma=ysigma,
                                   spike=False, transient=True)
        ## Clipping normalization
        # tuning_ = tf.normalize_tuning(tuning, prct=95)
        pvr_pool.append(pvcorr(tuning))
    pvr_list.append(np.hstack(pvr_pool))
    
# %% Plot for all datasets
def summary_pvr(data_list, varnames=['A','B','C'], varvalues=[None,None,None]):
    
    n_exp = len(data_list)
    df = ut.long_dataframe(data_list, varnames=varnames, varvalues=varvalues)
    struct = {'data': df, 'x': varnames[0], 'y': varnames[2], 'hue': varnames[1]}

    fig, ax = plt.subplots(figsize=(5.5,6))
    sns.boxplot(**struct)
    
    pformat = {'pvalue_thresholds': [[1e-3,'***'],[1e-2,'**'],[0.05,'*'],[1,'ns']],
                     'fontsize': 16}
    pairs, pvalues = [], []
    for p in range(n_exp):
        stat, pval = stats.kruskal(*data_list[p])
        print('-'*16 + varvalues[0][p] + '-'*16)
        print('Kruskal-Wallis test, p-value: %.4g' % pval)
        pval_paired = posthoc_dunn(data_list[p], p_adjust='bonferroni').to_numpy()
        print('Post-hoc Dunn\'s test')
        print('FF vs NN p-value: %.4g' % pval_paired[0,1])
        print('FF vs FN p-value: %.4g' % pval_paired[0,2])
        print('NN vs FN p-value: %.4g' % pval_paired[1,2])
        for u, v in zip([0,0,1],[1,2,2]):
            pairs.append(((varvalues[0][p],varvalues[1][u]),
                          (varvalues[0][p],varvalues[1][v])))
            pvalues.append(pval_paired[u,v])
    annot = Annotator(ax, pairs, **struct)
    annot.configure(test=None, loc='outside', line_width=1.2, line_height=0.01, pvalue_format=pformat)
    annot.set_pvalues(pvalues)
    annot.annotate()
    
    ax.set_xticklabels(varvalues[0], rotation=30)
    ax.set_xlabel('')
    ax.legend(loc='best')
    fig.tight_layout()
    
summary_pvr(pvr_list, varnames=['Experiment','Context','PV correlation'],
            varvalues=[exps,['FF','NN','FN'],None])

