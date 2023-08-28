# -*- coding: utf-8 -*-
"""
Take the invivo data file (*.hdf5) (see suite2p_workflow.py),
compute trial-to-trial similarity matrix and cluster trials.
Identify two clusters and decompose each trial into the two cluster means.

Short notations:
    m: mouse iterable (n_mice: number of animals)
    k: category iterable (ncat: number of categories/sessions)
    c: context iterable (nctx: number of context, 2 for Fam and Nov)
    i: cell iterable (ncell: total number of cells, ncells: number of cells per animal)
    
@author: Hung-Ling
"""
import os
from glob import glob
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering

os.chdir(r'C:\Users\Hung-Ling\Documents\Analysis_Python\MC_Analysis')
plt.style.use('figure.mplstyle')

from function.data2p import get_data_bis
import function.tuning_function as tf

# %% Analyze two clusters Fam/Nov
def compute_similarity(M):
    
    ntrial = M.shape[2]  # (ncell, ybin, ntrial)
    M_ = M.reshape((-1,ntrial)).T  # (ntrial, ncell*ybin)
    R = cosine_similarity(M_)  # Nonnegative affinity matrix (0 to 1)
        
    return R

def cluster_trials(R, context):
    '''
    Parameters
    ----------
    R : 2d array, shape (n_trials, n_trials)
        Similarity matrix used for clustering
    context : 1d array, shape (n_trials,)
        Context label for each trial

    Returns
    -------
    y : 1d array, shape (n_trials,)
        Cluster labels (majority class of familiar trials is labeled 0)
    '''
    # cluster = SpectralClustering(n_clusters=2, affinity='precomputed').fit(R)
    cluster = SpectralClustering(n_clusters=2, affinity='precomputed', assign_labels='cluster_qr').fit(R)
    y = cluster.labels_
    # l0 = stats.mode(y[np.where(context==0)[0]])[0].item()
    # if l0 == 0:
    if np.mean(y==context) > 0.5:
        return y
    else:
        return np.mod(y-1,2)

def decompose(X, y):
    
    reg = LinearRegression(fit_intercept=False).fit(X, y)
    l2 = lambda u: np.sqrt(np.sum(u**2))
    weight = reg.coef_/l2(reg.coef_)
    
    return weight.ravel()

def decompose_trial(M, tuning_cluster):
    
    ntrial = M.shape[2]
    X = np.column_stack([F.ravel() for F in tuning_cluster])  # (N_pixels, 2)
    weights = np.zeros((ntrial,2))
    for r in range(ntrial):
        y = M[:,:,r].reshape((-1,1))
        weights[r] = decompose(X, y)
    
    return weights
    
def analyze_clusters(M, context, prct=100):
    
    ntrial = M.shape[2]
    R = compute_similarity(M)
    
    ## Discard disconnected trials
    R2 = np.copy(R)
    np.fill_diagonal(R2, np.NaN)
    # bad_trial = np.where(np.nanmean(R2, axis=1)==0)[0]  # Equivalent to prct=100
    bad_trial = np.where(np.nanpercentile(R2, prct, axis=1)==0)[0]
    if len(bad_trial) > 0:
        good_trial = np.setdiff1d(np.arange(ntrial), bad_trial)
        context = context[good_trial]
        R = R[good_trial][:,good_trial]
        M = np.delete(M, bad_trial, axis=2)
    else:
        good_trial = np.arange(ntrial)
    
    label = cluster_trials(R, context)
    a = np.logical_and(context==0, label==0)
    b = np.logical_and(context==1, label==1)
    correct_class = np.array([a.sum()/np.sum(context==0), b.sum()/np.sum(context==1)])
    print('Proportion of correct class in fam: %.4g' % correct_class[0])
    print('Proportion of correct class in nov: %.4g' % correct_class[1])
    
    # Tuning averaged within the given context and cluster
    tuning_cluster = [M[:,:,a].mean(axis=2), M[:,:,b].mean(axis=2)]
    
    return R, label, good_trial, tuning_cluster

# %%
def draw_similarity(R, context, label, figsize=(5,5)):
    
    ntrial = len(R)
    
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.18, right=0.82, bottom=0.18, top=0.82)
    ax_ctx = fig.add_axes([0.18, 0.84, 0.64, 0.02])
    ax_lab = fig.add_axes([0.14, 0.18, 0.02, 0.64])
    cax = fig.add_axes([0.85, 0.18, 0.02, 0.64])
    
    cmap_ctx = colors.ListedColormap(['C0','C1'])  # plt.cm.Paired(1),plt.cm.Paired(7)
    cmap_lab = colors.ListedColormap(['C2','C3'])  # plt.cm.Paired(0),plt.cm.Paired(6)
    
    ax_ctx.axis('off')
    ax_ctx.imshow(context[np.newaxis,:], cmap=cmap_ctx)
    ax_ctx.set_aspect('auto')
    
    ax_lab.imshow(label[:,np.newaxis], cmap=cmap_lab, extent=[0,1,ntrial+0.5,0.5])
    ax_lab.set_aspect('auto')
    ax_lab.get_xaxis().set_visible(False)
    ax_lab.spines[['top','right','bottom']].set_visible(False)
    ax_lab.set_ylabel('Run')
    
    rs = R[np.triu_indices(R.shape[0], k=1)]
    # vmax = np.percentile(rs,99)
    # vmax = np.ceil(np.max(rs)*10)/10
    vmax = np.ceil(np.percentile(rs,99)*10)/10
    
    img = ax.imshow(R, vmin=0, vmax=vmax, cmap='viridis', extent=[0.5,ntrial+0.5]*2)
    ax.get_yaxis().set_visible(False)
    ax.spines[['top','right','left']].set_visible(False)
    ax.set_xlabel('Run')
    
    fig.colorbar(img, cax=cax)
    
    return fig, ax

# %% Setup
folder = r'D:\LW_Alldata\SI_Transient'
alldata = {'Enriched': glob(os.path.join(folder,'Enriched*.hdf5')),
           'Dissimilar': glob(os.path.join(folder,'Distinct*.hdf5')),
           'Similar': glob(os.path.join(folder,'Similar*.hdf5'))}
p = 0
exps = ['Enriched','Dissimilar','Similar']
datapath = alldata[exps[p]][3]
day = 1  # [1,2,3]  # Recording days, None for all days
min_rate = 0/60  # Minimum transient rate for active cells
ybin = 20  # Number of spatial bins
ysigma = 0  # Size of the Gaussian filter (in number of spatial bin) for smoothing the tuning curves

# %% Load data
data, cells, days, ctx, _ = get_data_bis(datapath, day=day, min_rate=min_rate, verbose=True)

with h5py.File(datapath, 'r') as f:
    yrange = f['params/yrange'][()]  # Dataset dependent
    
print('Recording days:', days)
print('Context fam/nov:', ctx)

tuning = tf.compute_tuning(data, yrange=yrange, ybin=ybin, ysigma=ysigma,
                           spike=False, transient=True)
M = np.concatenate(tuning, axis=2)  # Combine categories (ncell,ybin,ntrial)
context = np.hstack([np.repeat(ctx[k], tuning[k].shape[2])
                     for k in range(len(tuning))])

# %% Run for one mouse
R, label, good_trial, tuning_cluster = analyze_clusters(M, context, prct=90)
correct_class = np.sum(context[good_trial]==label)/len(label)
print('Proportion of correct class: %.4g' % correct_class)
weights = decompose_trial(M, tuning_cluster)

fig, ax = draw_similarity(R, context[good_trial], label, figsize=(4,4))  # (6,6)

# %% Check parameter
ybin_list = [4,6,8,10,20,30,40,60,80]
correct_list = np.zeros(len(ybin_list))

for i, ybin in enumerate(ybin_list):
    tuning = tf.compute_tuning(data, yrange=yrange, ybin=ybin, ysigma=ysigma,
                               spike=False, transient=True)
    M = np.concatenate(tuning, axis=2)  # Combine categories (ncell,ybin,ntrial)
    context = np.hstack([np.repeat(ctx[k], tuning[k].shape[2])
                         for k in range(len(tuning))])
    R, label, good_trial, tuning_cluster = analyze_clusters(M, context, prct=90)
    correct_list[i] = np.sum(context[good_trial]==label)/len(label)
    
fig, ax = plt.subplots(figsize=(5,4))
ax.plot(ybin_list, correct_list)
fig.tight_layout()

# %% Run for all datasets
day = 1  # Recording days, None for all days
min_rate = 0/60  # Minimum transient rate for active cells
ybin = 20  # Number of spatial bins
ysigma = 0  # Size of the Gaussian filter (in number of spatial bin) for smoothing the tuning curves
exps = ['Enriched','Dissimilar','Similar']
correct_list = [[] for _ in range(len(exps))]

for p, exp in enumerate(exps):
    print('Get %s datasets ...' % exp)
    for datapath in alldata[exp]:
        data, cells, days, ctx, _ = get_data_bis(datapath, day=day, min_rate=min_rate)
        with h5py.File(datapath, 'r') as f:
            yrange = f['params/yrange'][()]  # Dataset dependent
        tuning = tf.compute_tuning(data, yrange=yrange, ybin=ybin, ysigma=ysigma,
                                   spike=False, transient=True)
        M = np.concatenate(tuning, axis=2)  # Combine categories (ncell,ybin,ntrial)
        context = np.hstack([np.repeat(ctx[k], tuning[k].shape[2])
                             for k in range(len(tuning))])

        R, label, good_trial, tuning_cluster = analyze_clusters(M, context, prct=90)
        correct_class = np.sum(context[good_trial]==label)/len(label)
        correct_list[p].append(correct_class)
        
# %% Summary plot
exps = ['Enriched','Dissimilar','Similar']
palette = ['C3','C0','C2']
df_list = []
for p, exp in enumerate(exps):
    n_mice = len(correct_list[p])
    df_list.append(
        pd.DataFrame({'Fraction': correct_list[p],
                      'Context': [exp]*n_mice}))
    print('-'*16 + f' {exp} ' + '-'*16)
    pval = stats.shapiro(correct_list[p])[1]
    if pval >= 0.05:
        print(f'Shapiro test p={pval:.4g}, likely normal distribution')
    else:
        print(f'Shapiro test p={pval:.4g}, unlikely normal distribution')
    pval = stats.ttest_1samp(correct_list[p], 0.5, alternative='greater')[1]
    print(f't-test for fraction not greater than 0.5: p={pval:.4g}')
    # pval = stats.mannwhitneyu(correct_list[p], 0.5*np.ones(n_mice), alternative='greater')[1]
    # print(f'Mann-Whitney test for fraction not greater than 0.5: p={pval:.4g}')

df = pd.concat(df_list, ignore_index=True)

fig, ax = plt.subplots(figsize=(3.5,5))
ax.axhline(0.5, c='gray', lw=1.5, ls='--')
sns.stripplot(data=df, x='Context', y='Fraction', palette=palette, ax=ax, size=8)
sns.pointplot(data=df, x='Context', y='Fraction', palette=palette, ax=ax,
              errorbar='se', markers='x', capsize=0.25, join=False)
ax.set_xticklabels(exps, rotation=30)
ax.set_ylabel('Fraction correct')
ax.set_xlabel('')
fig.tight_layout()

with pd.ExcelWriter('Cluster.xlsx') as writer:
    df.to_excel(writer, sheet_name='Cluster', index=False)
    