# -*- coding: utf-8 -*-
"""
Take the invivo data file (*.hdf5) (see suite2p_workflow.py),
decode the position from spatial tuning data.

@author: Hung-Ling
"""
import os
from glob import glob
import numpy as np
import h5py
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from statannotations.Annotator import Annotator

os.chdir(r'C:\Users\Hung-Ling\Documents\Analysis_Python\MC_Analysis')
plt.style.use('figure.mplstyle')

from function.data2p import get_data_bis
import function.tuning_function as tf
import function.utils as ut
from decoder.manifold import SpatialDecoder
from decoder.metrics import get_error

# %% Useful functions
def bin_time_series(F, fps, dt, statistic='sum'):
    
    if F.ndim == 1:
        F = F.reshape((1,-1))
    ncell, T = F.shape
    t = T/fps  # Total time in s
    tbin = int(t/dt)  # Number of time bins
    ts = np.arange(T)/fps
    F2 = np.zeros((ncell, tbin))
    for i in range(ncell):
        F2[i], _, _ = stats.binned_statistic(
            ts, F[i], statistic=statistic, bins=tbin, range=(-0.5/fps, (T-0.5)/fps))
            
    return F2.squeeze()
    
def collect_data(data, dt=0.25, tsigma=0, ctx=np.array([0,1,0,1])):
    
    nctx = len(set(ctx))
    Fdata = [[] for _ in range(nctx)]  # Fluorescence
    ydata = [[] for _ in range(nctx)]  # Position
    
    for k, c in enumerate(ctx):  # k = category ; c = context
        ntrial = len(data['t'][k])
        nframes = np.hstack([0, np.cumsum(data['t'][k])])
        for r in range(ntrial):
            idx = slice(nframes[r], nframes[r+1])
            moving = data['moving'][k][idx].astype(bool)
            Ftmp = data['Tr'][k][:,idx][:,moving].astype(float)
            Ftmp[Ftmp<0] = 0  # Eliminate -0.0
            ytmp = data['y'][k][idx][moving].astype(float)
            ## Bin time points
            Ftmp = bin_time_series(Ftmp, data['fps'], dt, statistic='sum')
            ytmp = bin_time_series(ytmp, data['fps'], dt, statistic='mean')
            if tsigma > 0:
                Ftmp = gaussian_filter1d(Ftmp, tsigma, axis=1, mode='nearest')
            Fdata[c].append(Ftmp)
            ydata[c].append(ytmp)
        
    return Fdata, ydata

def prepare_data(Fs, ys, trial=None):
    '''
    Parameters
    ----------
    Fs : list (trial) of numpy array (ncell, tbin)
        Temporal activity of a given mouse with all trials collected in the list
    ys : list (trial) of numpy array (tbin,)
        Position vector of each trial
    trial : numpy 1d array
        Indices of selected trials

    Returns
    -------
    X : array (n_samples, ncell)
    y : array (n_samples,) 
        where n_samples = sum of tbin over trials
    '''
    if trial is None:  # Take all trials
        trial = np.arange(len(Fs))
    elif isinstance(trial, int):
        trial = [trial]   
    X = np.vstack([Fs[r].T for r in trial])
    y = np.hstack([ys[r] for r in trial])
    
    ## Discard points where all cells are silent
    silent = (X.sum(axis=1) == 0)
    if np.any(silent):
        X = X[~silent,:]
        y = y[~silent]
        
    return X, y

def shuffle_position(ys, seed=0):
    '''For each trial, circularly shuffle the position vector.
    '''
    rng = np.random.default_rng(seed)
    yshuffled = []
    for r in range(len(ys)):
        t = rng.choice(len(ys[r]))
        yshuffled.append(np.hstack([ys[r][t:], ys[r][:t]]))
    
    return yshuffled

def decode_position_cv(Fs, ys, nfold=10, shuffle=0, y2cm=100, **kwargs):
    
    error = []
    if shuffle:  # Non zero integer
        cv = KFold(n_splits=nfold, shuffle=True, random_state=shuffle)
    else:
        cv = KFold(n_splits=nfold, shuffle=False)
    
    for train_set, test_set in cv.split(np.zeros((len(Fs),1))): 
    
        X_train, y_train = prepare_data(Fs, ys, trial=train_set)
        X_test, y_test = prepare_data(Fs, ys, trial=test_set)
        
        spd = SpatialDecoder(**kwargs)
        spd.fit(X_train, y_train)
        y_pred = spd.predict(X_test)
        error.append(y2cm*get_error(y_test, y_pred, kind='median'))
    
    return np.array(error)

# %% Position decoder
def decode_position(datapath, day=1, min_rate=1/60, dt=0.25, tsigma=1, nfold=10,
                    shuffle=0, seed=0, pval_thr=0.05, **kwargs):
    ## Load data
    data, cells, days, ctx, _ = get_data_bis(
        datapath, day=day, min_rate=min_rate, verbose=False)

    ## Collect data (time series) by context
    Fdata, ydata = collect_data(data, dt=dt, tsigma=tsigma, ctx=ctx)
    nctx = len(set(ctx))
    
    # significant = (cells['si_pvalue'] <= pval_thr).T  # (nctx, ncell) if one day
    # Fdata_bis = [Fdata[c][:,:,significant[c]] for c in range(nctx)]

    ## Decoding position
    error = [[] for _ in range(nctx)]
    
    for c in range(nctx):
        
        error[c].append(decode_position_cv(
            Fdata[c], ydata[c], nfold=nfold, shuffle=shuffle, **kwargs))
        
        ## Shuffle position
        yshuffled = shuffle_position(ydata[c], seed=seed)
        error[c].append(decode_position_cv(
            Fdata[c], yshuffled, nfold=nfold, shuffle=shuffle, **kwargs))
        ## Place cells only
        # error[c].append(decode_position_cv(
        #     Fdata_bis[c], ydata[c], nfold=nfold, yrange=yrange, ybin=ybin, **kwargs))
        
        error[c] = np.vstack(error[c])  # (2, nfold) array
    
    return error

# %% Setup (for one experiment)
folder = r'D:\LW_Alldata\SI_Transient'
alldata = {'Enriched': glob(os.path.join(folder,'Enriched*.hdf5')),
           'Dissimilar': glob(os.path.join(folder,'Distinct*.hdf5')),
           'Similar': glob(os.path.join(folder,'Similar*.hdf5'))}
p = 2
exps = ['Enriched','Dissimilar','Similar']
datasets = alldata[exps[p]]
day = 1  # Recording days, None for all days
min_rate = 1/60  # Minimum transient rate for active cells
dt = 0.25  # Time bin in s
tsigma = 2  # Size of the Gaussian filter (in number of temporal bin) for smoothing the activity trace

# %% Run decoder 
params = dict(method='knr', k=10, knr_metric='cosine', reduction='none')  # 'knr'|'gpr'|'svr'|'pvcorr'|'bayesian
error_list = []

for m, datapath in enumerate(datasets):
    
    print('Decoding mouse %d ...' % (m+1))
    error = decode_position(datapath, day=day, min_rate=min_rate, dt=dt, tsigma=tsigma,
                            nfold=10, shuffle=2, seed=2, **params)
    error_list.append(error)
    df = ut.long_dataframe(error, varnames=['Context','Data','Error'],
                           varvalues=[['Familiar','Novel'],
                                      ['Original','Shuffle trial','Shuffle position'],  # 'Place cells'
                                      None])
    fig, ax = plt.subplots(figsize=(4,5))
    sns.barplot(df, x='Context', y='Error', hue='Data', ax=ax)
    ax.set(ylabel='Error (cm)')
    ax.legend(loc='best')
    fig.tight_layout()
    
# %% Summary
def plot_summary(error_list, color='C0'):
    
    summary = [[] for _ in range(len(error_list))]
    for m, error in enumerate(error_list):  # error: list (context) of array, shape (3, nfold)
        summary[m] = np.vstack([np.mean(err, axis=1) for err in error])
    summary = np.vstack(summary)
    df_summary = pd.DataFrame(
        summary, columns=['Original','Shuffle position'])  # 'Place cells'
    
    fig, ax = plt.subplots(figsize=(4,5))
    sns.barplot(df_summary, errorbar='se', width=0.6, capsize=0.3, linewidth=2, errwidth=2,
                errcolor=color, edgecolor=color, facecolor=(0,)*4)
    for point in summary:
        ax.plot(range(len(point)), point, c='gray', alpha=0.8, lw=1.2,
                marker='o', ms=6, mfc='none')
        
    pformat = {'pvalue_thresholds': [[1e-3,'***'],[1e-2,'**'],[0.05,'*'],[1,'ns']]}
    pairs = [('Original','Shuffle position')]  # ('Original','Place cells')
    annot = Annotator(ax, pairs, data=df_summary)
    annot.configure(test='t-test_paired', loc='inside', pvalue_format=pformat)  # Wilcoxon t-test_paired
    annot.apply_and_annotate()
    
    ax.set(ylabel='Error (cm)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    fig.tight_layout()

plot_summary(error_list, color='tab:red')

# %% Importance of each cell
def cell_importance(Fdata, ydata, nfold=10, shuffle=0, **kwargs):
    
    importance = [[] for _ in range(len(Fdata))]
    
    for c in range(len(Fdata)):
        print('Decoding context %d ...' % c)
        error = decode_position_cv(
            Fdata[c], ydata[c], nfold=nfold, shuffle=shuffle, **kwargs)
        
        ntrial = len(Fdata[c])
        ncell = Fdata[c][0].shape[0]
        error_cell = [[] for _ in range(ncell)]
        for i in range(ncell):
            ind = np.setdiff1d(np.arange(ncell), i)
            F = [Fdata[c][r][ind,:] for r in range(ntrial)]  # Remove the i-th cell
            error_cell[i] = decode_position_cv(
                F, ydata[c], nfold=nfold, shuffle=shuffle, **kwargs)
        error_cell = np.vstack(error_cell)  # (ncell, ntrial)
        importance[c] = np.mean(error_cell, axis=1) - np.mean(error)
        importance[c] /= np.std(importance[c])
    
    return np.column_stack(importance)

def decode_position_importance(datapath, day=1, min_rate=1/60, dt=0.25, tsigma=0, 
                               nfold=10, shuffle=0, **kwargs):
    ## Load data
    data, cells, days, ctx, _ = get_data_bis(
        datapath, day=day, min_rate=min_rate, verbose=False)
    
    ## Collect data (time series) by context
    Fdata, ydata = collect_data(data, dt=dt, tsigma=tsigma, ctx=ctx)
    
    importance = cell_importance(
        Fdata, ydata, nfold=nfold, shuffle=shuffle, **kwargs)  # (ncell, nctx)

    return importance

params = dict(method='knr', k=10, knr_metric='cosine', reduction='none')  # 'knr'|'gpr'|'svr'|'pvcorr'|'bayesian
importance_list = []

for m, datapath in enumerate(datasets):
    
    print('Decoding mouse %d ...' % (m+1))
    imp = decode_position_importance(datapath, day=day, min_rate=min_rate,
                                     dt=dt, tsigma=tsigma, nfold=10, shuffle=2,
                                     **params)
    importance_list.append(imp)

# %% Save results
from PyQt5.QtWidgets import QFileDialog

def save_results(error_list, importance_list):
    
    savename = QFileDialog.getSaveFileName(filter='XLSX (*.xlsx)')[0]
    n_mice = len(error_list)
    df_list = []
    
    for error in error_list:
        df_list.append(pd.DataFrame(
            np.vstack(error).T, 
            columns=['F','F shuffle position',
                     'N','N shuffle position']))
    
    df = pd.DataFrame(np.vstack(importance_list), columns=['Fam','Nov'])
    
    with pd.ExcelWriter(savename) as writer:
        for m in range(n_mice):
            df_list[m].to_excel(writer, sheet_name=str(m+1), index=False)
        df.to_excel(writer, sheet_name='Importance', index=False)
        
save_results(error_list, importance_list)



