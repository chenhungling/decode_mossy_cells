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
def collect_data(tuning, ctx=np.array([0,1,0,1]), ybin=80):
    
    y_edges = np.linspace(0,4,ybin+1)  # Unify position data to (0,4)
    y_centers = (y_edges[:-1] + y_edges[1:])/2
    Fdata, ydata  = [], []
    for c in range(len(set(ctx))):
        F = np.concatenate([tuning[k] for k in np.where(ctx==c)[0]], axis=2)  # (ncell, ybin, ntrial)
        Fdata.append(np.transpose(F, (2,1,0)))  # (ntrial, ybin, ncell)
        ydata.append(np.tile(y_centers, (F.shape[2],1)))  # (ntrial, ybin)
        
    return Fdata, ydata

def prepare_data(Fs, ys, trial=None):
    '''
    Parameters
    ----------
    Fs : numpy array (ntrial, ybin, ncell)
        Spatial tuning of a given mouse with all trials stacked along axis=0
    ys : numpy array (ntrial, ybin)
        Position vector of each trial
    trial : numpy 1d array
        Indices of selected trials

    Returns
    -------
    X : array (n_samples, ncell)
    y : array (n_samples,) 
        where n_samples = ntrial * ybin
    '''
    if trial is None:  # Take all trials
        trial = np.arange(Fs.shape[0])
    elif isinstance(trial, int):
        trial = [trial]   
    X = np.vstack([Fs[r,:,:] for r in trial])
    y = np.hstack([ys[r,:] for r in trial])
    
    ## Discard points where all cells are silent
    silent = (X.sum(axis=1) == 0)
    if np.any(silent):
        X = X[~silent,:]
        y = y[~silent]
        
    return X, y

def shuffle_trial(Fs, seed=0):
    '''For each cell, shuffle the trial ID while keep the spatial tuning.
    '''
    rng = np.random.default_rng(seed)
    Fshuffled = Fs.copy()  # (ntrial, ybin, ncell)
    for i in range(Fs.shape[2]):
        Fshuffled[:,:,i] = rng.permutation(Fs[:,:,i])  # Shuffle by rows (trials)
        
    return Fshuffled

def shuffle_position(ys, seed=0):
    '''For each trial, circularly shuffle the position vector.
    '''
    rng = np.random.default_rng(seed)
    yshuffled = ys.copy()
    for r in range(ys.shape[0]):
        t = rng.choice(ys.shape[1])
        yshuffled[r,:] = np.hstack([ys[r,t:], ys[r,:t]])
    
    return yshuffled

def decode_position_cv(Fs, ys, nfold=10, shuffle=0, y2cm=100, **kwargs):
    
    error = []
    if shuffle:  # Non zero integer
        cv = KFold(n_splits=nfold, shuffle=True, random_state=shuffle)
    else:
        cv = KFold(n_splits=nfold, shuffle=False)
    
    for train_set, test_set in cv.split(np.zeros((Fs.shape[0],1))): 
    
        X_train, y_train = prepare_data(Fs, ys, trial=train_set)
        X_test, y_test = prepare_data(Fs, ys, trial=test_set)
        
        spd = SpatialDecoder(**kwargs)
        spd.fit(X_train, y_train)
        y_pred = spd.predict(X_test)
        error.append(y2cm*get_error(y_test, y_pred, kind='median'))
    
    return np.array(error)

# %% Position decoder
def decode_position(datapath, day=1, min_rate=1/60, ybin=80, ysigma=1, nfold=10,
                    shuffle=0, seed=0, pval_thr=0.05, **kwargs):
    ## Load data
    data, cells, days, ctx, _ = get_data_bis(
        datapath, day=day, min_rate=min_rate, verbose=False)
    
    with h5py.File(datapath, 'r') as f:
        yrange = f['params/yrange'][()]  # Dataset dependent

    
    
    ## Prepare data (spatial tuning)
    tuning = tf.compute_tuning(data, yrange=yrange, ybin=ybin, ysigma=ysigma,
                               spike=False, transient=True)
    ## Collect data by context
    Fdata, ydata = collect_data(tuning, ctx=ctx, ybin=ybin)
    nctx = len(set(ctx))
    
    # significant = (cells['si_pvalue'] <= pval_thr).T  # (nctx, ncell) if one day
    # Fdata_bis = [Fdata[c][:,:,significant[c]] for c in range(nctx)]

    ## Decoding position
    error = [[] for _ in range(nctx)]
    
    for c in range(nctx):
        
        error[c].append(decode_position_cv(Fdata[c], ydata[c], nfold=nfold, shuffle=shuffle,
                                           yrange=yrange, ybin=ybin, **kwargs))
        ## Shuffle trials
        Fshuffled = shuffle_trial(Fdata[c], seed=seed)
        error[c].append(decode_position_cv(Fshuffled, ydata[c], nfold=nfold, shuffle=shuffle,
                                           yrange=yrange, ybin=ybin, **kwargs))
        ## Shuffle position
        yshuffled = shuffle_position(ydata[c], seed=seed)
        error[c].append(decode_position_cv(Fdata[c], yshuffled, nfold=nfold, shuffle=shuffle,
                                           yrange=yrange, ybin=ybin, **kwargs))
        ## Place cells only
        # error[c].append(decode_position_cv(
        #     Fdata_bis[c], ydata[c], nfold=nfold, yrange=yrange, ybin=ybin, **kwargs))
        
        error[c] = np.vstack(error[c])  # (4, nfold) array
    
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
min_rate = 0/60  # Minimum transient rate for active cells
ybin = 80  # Number of spatial bins
ysigma = 2  # Size of the Gaussian filter (in number of spatial bin) for smoothing the tuning curves

# %% Run decoder 
params = dict(method='knr', k=10, knr_metric='cosine', reduction='none')  # 'knr'|'gpr'|'svr'|'pvcorr'|'bayesian
error_list = []

for m, datapath in enumerate(datasets):
    
    print('Decoding mouse %d ...' % (m+1))
    error = decode_position(datapath, day=day, min_rate=min_rate, ybin=ybin,
                            ysigma=ysigma, nfold=10, shuffle=2, seed=2, **params)
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
        summary, columns=['Original','Shuffle trial','Shuffle position'])  # 'Place cells'
    
    fig, ax = plt.subplots(figsize=(4,5))
    sns.barplot(df_summary, errorbar='se', width=0.6, capsize=0.3, linewidth=2, errwidth=2,
                errcolor=color, edgecolor=color, facecolor=(0,)*4)
    for point in summary:
        ax.plot(range(len(point)), point, c='gray', alpha=0.8, lw=1.2,
                marker='o', ms=6, mfc='none')
        
    pformat = {'pvalue_thresholds': [[1e-3,'***'],[1e-2,'**'],[0.05,'*'],[1,'ns']]}
    pairs = [('Original','Shuffle trial'), ('Original','Shuffle position')]  # ('Original','Place cells')
    annot = Annotator(ax, pairs, data=df_summary)
    annot.configure(test='t-test_paired', loc='inside', pvalue_format=pformat)  # Wilcoxon t-test_paired
    annot.apply_and_annotate()
    
    ax.set(ylabel='Error (cm)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    fig.tight_layout()

plot_summary(error_list, color='tab:red')

# %% Importance of each cell
def cell_importance(Fdata, ydata, nfold=10, shuffle=0, yrange=(0,4), ybin=80, **kwargs):
    
    importance = [[] for _ in range(len(Fdata))]
    
    for c in range(len(Fdata)):
        print('Decoding context %d ...' % c)
        error = decode_position_cv(
            Fdata[c], ydata[c], nfold=nfold, shuffle=shuffle, yrange=yrange, ybin=ybin, **kwargs)
    
        ncell = Fdata[c].shape[2]
        error_cell = [[] for _ in range(ncell)]
        for i in range(ncell):
            ind = np.setdiff1d(np.arange(ncell), i)
            F = Fdata[c][:,:,ind]  # Remove the i-th cell
            error_cell[i] = decode_position_cv(
                F, ydata[c], nfold=nfold, shuffle=shuffle, yrange=yrange, ybin=ybin, **kwargs)
        error_cell = np.vstack(error_cell)  # (ncell, ntrial)
        importance[c] = np.mean(error_cell, axis=1) - np.mean(error)
        importance[c] /= np.std(importance[c])
    
    return np.column_stack(importance)

def decode_position_importance(datapath, day=1, min_rate=1/60, ybin=80, ysigma=1, 
                               nfold=10, shuffle=0, **kwargs):
    ## Load data
    data, cells, days, ctx, _ = get_data_bis(
        datapath, day=day, min_rate=min_rate, verbose=False)
    
    with h5py.File(datapath, 'r') as f:
        yrange = f['params/yrange'][()]  # Dataset dependent
    
    ## Prepare data (spatial tuning)
    tuning = tf.compute_tuning(data, yrange=yrange, ybin=ybin, ysigma=ysigma,
                               spike=False, transient=True)

    ## Collect data by context
    Fdata, ydata = collect_data(tuning, ctx=ctx, ybin=ybin)
    
    importance = cell_importance(
        Fdata, ydata, nfold=nfold, shuffle=shuffle, yrange=yrange, ybin=ybin, **kwargs)  # (ncell, nctx)

    return importance

params = dict(method='knr', k=10, knr_metric='cosine', reduction='none')  # 'knr'|'gpr'|'svr'|'pvcorr'|'bayesian
importance_list = []

for m, datapath in enumerate(datasets):
    
    print('Decoding mouse %d ...' % (m+1))
    imp = decode_position_importance(datapath, day=day, min_rate=min_rate, 
                                     nfold=10, shuffle=2, ysigma=ysigma, ybin=ybin,
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
            columns=['F','F shuffle trial','F shuffle position',
                     'N','N shuffle trial','N shuffle position']))
    
    df = pd.DataFrame(np.vstack(importance_list), columns=['Fam','Nov'])
    
    with pd.ExcelWriter(savename) as writer:
        for m in range(n_mice):
            df_list[m].to_excel(writer, sheet_name=str(m+1), index=False)
        df.to_excel(writer, sheet_name='Importance', index=False)
        
save_results(error_list, importance_list)



