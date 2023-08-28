# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 11:33:09 2022

@author: Hung-Ling
"""
import os
from glob import glob
import numpy as np
import h5py
from scipy import stats
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.anova import AnovaRM

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

# %% Functions for decoding context
def prepare_data(M, label, trial=None):
    
    ntrial, ybin, ncell = M.shape
    
    if trial is None:  # Take all trials
        trial = np.arange(ntrial)
    elif isinstance(trial, int):
        trial = [trial]
        
    X = np.vstack([M[r].ravel() for r in trial])  # Stack ybin as features
    y = label[trial]
    
    return X, y

def decode_context_cv(M, label, nfold=10, shuffle=0):
    
    ntrial, ybin, ncell = M.shape
    accuracy = []
    coefficient = []
    if shuffle:  # Non zero integer
        cv = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=shuffle)
    else:
        cv = StratifiedKFold(n_splits=nfold, shuffle=False)
    
    for train_set, test_set in cv.split(np.zeros((ntrial,1)), y=label):  # X does not matter here 
    
        X_train, y_train = prepare_data(M, label, trial=train_set)
        X_test, y_test = prepare_data(M, label, trial=test_set)
        
        ## Linear support-vector classifier
        clf = SVC(kernel='linear')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        for r in range(len(y_test)):
            accuracy.append(y_pred[r]==y_test[r])
            
        coefficient.append(np.reshape(clf.coef_, (ybin, ncell)))
        
    return np.array(accuracy), np.stack(coefficient).mean(axis=0).T  # (ncell, ybin)

def decode_context_mouse(datapath, day=[1,2,3], min_rate=1/60, ybin=80, ysigma=1,
                         nfold=10, shuffle=0):
    
    data, cells, days, ctx, _ = get_data_bis(datapath, day=day, min_rate=min_rate)
    with h5py.File(datapath, 'r') as f:
        yrange = f['params/yrange'][()]  # Dataset dependent
        
    tuning = tf.compute_tuning(data, yrange=yrange, ybin=ybin, ysigma=ysigma,
                               spike=False, transient=True)
    ncell = tuning[0].shape[0]
    nday = len(day)
    accuracy = np.zeros(nday)
    weight = np.zeros((ncell,nday))
    
    for d in range(nday):
        sess = np.where(days==(d+1))[0]
        M = np.concatenate([tuning[s] for s in sess], axis=2)  # (ncell, ybin, ntrial)
        M = np.transpose(M, (2,1,0))  # (ntrial, ybin, ncell)
        label = np.hstack([np.repeat(ctx[k], tuning[k].shape[2]) for k in sess])  # (ntrial,)    
        
        accu, coeff = decode_context_cv(M, label, nfold=nfold, shuffle=shuffle)
        accuracy[d] = np.mean(accu)
        coeff_mean = np.mean(coeff, axis=1)  # Average across spatial bin
        weight[:,d] = coeff_mean/np.std(coeff_mean)  # Standardize each mouse
        
    return accuracy, weight
    
# %% Decoding context along days
day = [1,2,3]  # Recording days, None for all days
min_rate = 0/60  # Minimum transient rate for active cells
ybin = 20  # Number of spatial bins
ysigma = 0  # Size of the Gaussian filter (in number of spatial bin) for smoothing the tuning curves
nfold = 10
shuffle = 2

df_list = []
df_bis_list = []
nday = len(day)

for p, exp in enumerate(exps):
    
    print(f'Processing {exp} datasets ...')
    n_mice = len(alldata[exp])
    
    accuracy = [[] for _ in range(n_mice)]
    weight = [[] for _ in range(n_mice)]
    session = [[] for _ in range(n_mice)]
    mouse = [[] for _ in range(n_mice)]
    
    for m, datapath in enumerate(alldata[exp]):
        
        print('Calculating for mouse %d' % (m+1))
        accuracy[m], weight[m] = decode_context_mouse(
            datapath, day=day, min_rate=min_rate, ybin=ybin, ysigma=ysigma,
            nfold=nfold, shuffle=shuffle)
        
        session[m] = np.arange(1,nday+1)
        mouse[m] = np.repeat(m, nday)
    
    df = pd.DataFrame({'Accuracy': np.hstack(accuracy),
                       'Day': np.hstack(session),
                       'Mouse': np.hstack(mouse)})
    df_bis = pd.DataFrame(np.vstack(weight), columns=day)
    df_list.append(df)
    df_bis_list.append(df_bis)
    
    res = AnovaRM(data=df, depvar='Accuracy', subject='Mouse', within=['Day']).fit()
    print(res.anova_table)
        
    fig, ax = plt.subplots(figsize=(3,4))
    sns.lineplot(df, x='Day', y='Accuracy', lw=2, errorbar='se', 
                 err_style='bars', err_kws=dict(elinewidth=2))
    fig.tight_layout()

# %%
weight = df_bis_list[2].to_numpy().T

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(*weight, c='dimgray')
ax.plot(weight[0], weight[1], 'g.', alpha=0.5, zdir='z', zs=-4)
ax.plot(weight[1], weight[2], 'r.', alpha=0.5, zdir='x', zs=-4)

print('Day1-2 correlation r=%.4g p=%.4g' % stats.pearsonr(weight[0], weight[1]))
print('Day2-3 correlation r=%.4g p=%.4g' % stats.pearsonr(weight[1], weight[2]))

# %%
from PyQt5.QtWidgets import QFileDialog

def save_results(df_list, df_bis_list, exps=['Enriched','Dissimilar','Similar']):
    
    savename = QFileDialog.getSaveFileName(filter='XLSX (*.xlsx)')[0]
    
    with pd.ExcelWriter(savename) as writer:
        for p in range(len(exps)):
            df_list[p].to_excel(writer, sheet_name=exps[p], index=False)
            df_bis_list[p].to_excel(writer, sheet_name=exps[p]+'_weight', index=False)

save_results(df_list, df_bis_list, exps=exps)