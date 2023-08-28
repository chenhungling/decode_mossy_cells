# -*- coding: utf-8 -*-
"""
Take 2-photon simple data structure (see get_mat_data.py),
analyze cells properties along days.

@author: Hung-Ling
"""
import os
from glob import glob
import numpy as np
import h5py
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir(r'C:\Users\Hung-Ling\Documents\Analysis_Python\MC_Analysis')
plt.style.use('figure.mplstyle')

from function.data2p import get_data_bis
import function.tuning_function as tf

# %% Setup
folder = r'D:\LW_Alldata\SI_Transient'
alldata = {'Enriched': glob(os.path.join(folder,'Enriched*.hdf5')),
           'Dissimilar': glob(os.path.join(folder,'Distinct*.hdf5')),
           'Similar': [os.path.join(folder,'Similar5_937_invivo.hdf5'),
                       os.path.join(folder,'Similar6_939_invivo.hdf5'),
                       os.path.join(folder,'Similar7_948_invivo.hdf5'),
                       os.path.join(folder,'Similar8_5454_invivo.hdf5')]}
exps = ['Enriched','Dissimilar','Similar']
colors = ['tab:red','tab:blue','tab:green']

# %% Activity difference across day
day = [1,2,3]  # Recording days, None for all days
min_rate = 1/60  # Minimum transient rate for active cells
nday = len(day)
dauc_list = []

for p, exp in enumerate(exps):

    datapaths = alldata[exp]
    dauc = []
    for m, datapath in enumerate(datapaths):
        data, cells, days, ctx, _ = get_data_bis(datapath, day=day, min_rate=min_rate)
        auc = 60*cells['trate'].T  # (2*nday, ncell)
        dauc.append(np.vstack([auc[2*d+1]-auc[2*d] for d in range(nday)]))  # (nday, ncell) Nov-Fam
        # dauc.append(np.vstack([(auc[2*d+1]-auc[2*d])/(auc[2*d+1]+auc[2*d]) for d in range(nday)]))
    dauc = np.hstack(dauc)    
    dauc = dauc[:,np.all(np.isfinite(dauc),axis=0)]
    dauc_list.append(dauc)
    
    print('-'*16 + exp + '-'*16)
    print('Day1-2 correlation r=%.4g p=%.4g' % stats.pearsonr(dauc[0], dauc[1]))
    print('Day2-3 correlation r=%.4g p=%.4g' % stats.pearsonr(dauc[1], dauc[2]))

# %%
fig, axs = plt.subplots(2,3,figsize=(6.8,5.4))
for p, dauc in enumerate(dauc_list):
    df12 = pd.DataFrame(dauc[0:2].T, columns=['D1','D2'])
    sns.regplot(df12, x='D1', y='D2', ax=axs[0,p], color=colors[p], scatter_kws=dict(s=8), line_kws=dict(lw=2))
    df23 = pd.DataFrame(dauc[1:3].T, columns=['D2','D3'])
    sns.regplot(df23, x='D2', y='D3', ax=axs[1,p], color=colors[p], scatter_kws=dict(s=8), line_kws=dict(lw=2))
for ax in axs.ravel():
    ax.set(xlabel='', ylabel='')
axs[0,0].set(ylabel='Day 2')
axs[1,0].set(ylabel='Day 3')
axs[0,1].set(xlabel='Day 1 activity difference (transient/min)')
axs[1,1].set(xlabel='Day 2 activity difference (transient/min)')
fig.tight_layout()
fig.subplots_adjust(hspace=0.36, wspace=0.24)

# %% Spatial information across day
day = [1,2,3]  # Recording days, None for all days
min_rate = 1/60  # Minimum transient rate for active cells
nday = len(day)
si_list = []

for p, exp in enumerate(exps):
    
    datapaths = alldata[exp]
    si_nov = []
    for m, datapath in enumerate(datapaths):
        data, cells, days, ctx, _ = get_data_bis(datapath, day=day, min_rate=min_rate)
        si_nov.append(cells['si_unbiased'][:,1::2].T)  # Nov
    si_nov = np.hstack(si_nov)
    si_nov = si_nov[:,np.all(np.isfinite(si_nov),axis=0)]
    si_list.append(si_nov)
    
    print('-'*16 + exp + '-'*16)
    print('Day1-2 correlation r=%.4g p=%.4g' % stats.pearsonr(si_nov[0], si_nov[1]))
    print('Day2-3 correlation r=%.4g p=%.4g' % stats.pearsonr(si_nov[1], si_nov[2]))

# %%
fig, axs = plt.subplots(2,3,figsize=(6.8,5.4))
for p, si in enumerate(si_list):
    df12 = pd.DataFrame(si[0:2].T, columns=['D1','D2'])
    sns.regplot(df12, x='D1', y='D2', ax=axs[0,p], color=colors[p], scatter_kws=dict(s=8), line_kws=dict(lw=2))
    df23 = pd.DataFrame(si[1:3].T, columns=['D2','D3'])
    sns.regplot(df23, x='D2', y='D3', ax=axs[1,p], color=colors[p], scatter_kws=dict(s=8), line_kws=dict(lw=2))
for ax in axs.ravel():
    ax.set(xlabel='', ylabel='')
axs[0,0].set(ylabel='Day 2')
axs[1,0].set(ylabel='Day 3')
axs[0,1].set(xlabel='Day 1 spatial information ($\sigma$)')
axs[1,1].set(xlabel='Day 2 spatial information ($\sigma$)')
fig.tight_layout()
fig.subplots_adjust(hspace=0.36, wspace=0.24)

# %% Pair correlation across day
def calculate_pcorr(datapath, day=[1,2,3], min_rate=1/60, ybin=1, ysigma=0):
    
    data, cells, days, ctx, _ = get_data_bis(datapath, day=day, min_rate=min_rate)
    with h5py.File(datapath, 'r') as f:
        yrange = f['params/yrange'][()]  # Dataset dependent
    ncell = data['F'][0].shape[0]
    activity = tf.compute_tuning(data, yrange=yrange, ybin=ybin, ysigma=ysigma,
                                 spike=False, transient=True)
    
    Pcorr = []
    nday = len(day)
    nctx = len(set(ctx))
    for d in range(nday):
        for c in range(nctx):
            ks = np.where((days==(d+1)) & (ctx==c))[0]
            M = np.concatenate([activity[k] for k in ks], axis=2)  # (ncell,ybin,ntrial)
    # for k in range(len(activity)):
    #     M = activity[k]
            M = np.transpose(M, [2,1,0])  # (ntrial,ybin,ncell)
            A = np.reshape(M, (-1,ncell)).T  # (ncell, ybin*ntrial)
            pcorr = np.ones((ncell,)*2)  # Pairwise correlation
            for i in range(ncell):
                for j in range(i+1,ncell):
                    pcorr[i,j] = stats.kendalltau(A[i], A[j])[0]
                    pcorr[j,i] = pcorr[i,j]
            Pcorr.append(pcorr[np.triu_indices(ncell, k=1)])
        
    Pcorr = np.vstack(Pcorr)  # (nday*nctx, N(N-1)/2) where N = ncell
    ind = np.all(np.isfinite(Pcorr), axis=0)
    
    return Pcorr[:,ind]

Pcorr = calculate_pcorr(alldata['Enriched'][2], min_rate=0, ybin=1)

# N = 3*4
N = 3*2
order = np.hstack([np.arange(N)[0::2], np.arange(N)[1::2]])

fig, axs = plt.subplots(1,2)
axs[0].imshow(np.corrcoef(Pcorr))
axs[1].imshow(np.corrcoef(Pcorr[order]))

# %%
p = 1
datapaths = alldata[exps[p]]

R_pcorr = []
for m, datapath in enumerate(datapaths):
    print('Computing for mouse %d ...' % (m+1))
    Pcorr = calculate_pcorr(datapath, min_rate=0, ybin=1)
    R_pcorr.append(np.corrcoef(Pcorr))
    
R_mean = np.mean(np.stack(R_pcorr), axis=0)

N = 3*2
order = np.hstack([np.arange(N)[0::2], np.arange(N)[1::2]])

fig, axs = plt.subplots(1,2)
axs[0].imshow(R_mean)
axs[1].imshow(R_mean[order][:,order])
