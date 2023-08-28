# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 17:43:04 2022

@author: Hung-Ling
"""
import os
from glob import glob
import numpy as np
import h5py
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.anova import AnovaRM

os.chdir(r'C:\Users\Hung-Ling\Documents\Analysis_Python\MC_Analysis')
plt.style.use('figure.mplstyle')

from function.data2p import get_data_bis
import function.tuning_function as tf
from decoder.manifold import SpatialDecoder
from decoder.metrics import get_error

# %% Setup
folder = r'D:\LW_Alldata\SI_Transient'
alldata = {'Enriched': glob(os.path.join(folder,'Enriched*.hdf5')),
           'Dissimilar': glob(os.path.join(folder,'Distinct*.hdf5')),
           'Similar': [os.path.join(folder,'Similar5_937_invivo.hdf5'),
                       os.path.join(folder,'Similar6_939_invivo.hdf5'),
                       os.path.join(folder,'Similar7_948_invivo.hdf5'),
                       os.path.join(folder,'Similar8_5454_invivo.hdf5')]}
exps = ['Enriched','Dissimilar','Similar']

# %% Functions for decoding position
def prepare_data(Fs, ys, trial=None):
    
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

def decode_position_mouse(datapath, day=[1,2,3], min_rate=1/60, ybin=80, ysigma=1,
                          nfold=10, shuffle=0, **kwargs):
    
    data, cells, days, ctx, _ = get_data_bis(datapath, day=day, min_rate=min_rate)
    with h5py.File(datapath, 'r') as f:
        yrange = f['params/yrange'][()]  # Dataset dependent
        
    tuning = tf.compute_tuning(data, yrange=yrange, ybin=ybin, ysigma=ysigma,
                               spike=False, transient=True)
    nday = len(day)
    errors = np.zeros((nday,2))
    
    for d in range(nday):
        for c in range(2):
            sess = np.where((days==(d+1)) & (ctx==c))[0]
            M = np.concatenate([tuning[s] for s in sess], axis=2)  # (ncell, ybin, ntrial)
            Fs = np.transpose(M, (2,1,0))  # (ntrial, ybin, ncell)
            y_edges = np.linspace(0,4, ybin+1)  # Unify position data to (0,4)
            y_centers = (y_edges[:-1] + y_edges[1:])/2
            ys = np.tile(y_centers, (Fs.shape[0],1))  # (ntrial, ybin)
            
            error = decode_position_cv(Fs, ys, nfold=nfold, shuffle=shuffle, **kwargs)
            errors[d,c] = np.mean(error)
        
    return errors

# %% Decoding position along days
day = [1,2,3]
min_rate = 0
ybin = 80  # Number of spatial bins
ysigma = 2  # Size of the Gaussian filter (in number of spatial bin) for smoothing the tuning curves
nfold = 10
shuffle = 2
params = dict(method='knr', k=10, reduction='none')  # 'knr'|'gpr'|'svr'|'pvcorr'|'bayesian

df_list = []
nday = len(day)

for p, exp in enumerate(exps):
    
    print(f'Processing {exp} datasets ...')
    n_mice = len(alldata[exp])
    errors = [[] for _ in range(n_mice)]
    session = [[] for _ in range(n_mice)]
    context = [[] for _ in range(n_mice)]
    mouse = [[] for _ in range(n_mice)]
    
    for m, datapath in enumerate(alldata[exp]):
        
        print('Calculating for mouse %d' % (m+1))
        error = decode_position_mouse(
            datapath, day=day, min_rate=min_rate, ybin=ybin, ysigma=ysigma, 
            nfold=nfold, shuffle=shuffle, **params)
        
        errors[m] = error.ravel()  # Day1 F/N, Day2 F/N, ...
        session[m] = np.repeat(np.arange(1,nday+1),2)
        context[m] = np.tile([0,1],nday)
        mouse[m] = np.repeat(m,2*nday)
        
    df = pd.DataFrame({'Error': np.hstack(errors),
                       'Day': np.hstack(session),
                       'Context': np.hstack(context),
                       'Mouse': np.hstack(mouse)})
    df_list.append(df)
    
    # res = AnovaRM(data=df, depvar='Error', subject='Mouse', within=['Day']).fit()
    # print(res.anova_table)
    
    fig, ax = plt.subplots(figsize=(3,4))
    sns.lineplot(df, x='Day', y='Error', hue='Context', lw=2, errorbar='se', 
                 err_style='bars', err_kws=dict(elinewidth=2))
    fig.tight_layout()

# %%
from PyQt5.QtWidgets import QFileDialog

def save_results(df_list, exps=['Enriched','Dissimilar','Similar']):
    
    savename = QFileDialog.getSaveFileName(filter='XLSX (*.xlsx)')[0]
    
    with pd.ExcelWriter(savename) as writer:
        for p in range(len(exps)):
            df_list[p].to_excel(writer, sheet_name=exps[p], index=False)

save_results(df_list, exps=exps)