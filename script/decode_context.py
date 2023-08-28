# -*- coding: utf-8 -*-
"""
Take the invivo data file (*.hdf5) (see suite2p_workflow.py),
decode the context variable using SVC and study the decoder weight of each cell.

@author: Hung-Ling
"""
import os
from glob import glob
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from statannotations.Annotator import Annotator

os.chdir(r'C:\Users\Hung-Ling\Documents\Analysis_Python\MC_Analysis')
plt.style.use('figure.mplstyle')

from function.data2p import get_data_bis
import function.tuning_function as tf
import function.utils as ut
from decoder.metrics import get_majority

# %% Useful functions
def prepare_data(M, label, trial=None):
    '''
    Parameters
    ----------
    M : numpy 3d array, shape (ntrial, ybin, ncell)
        Spatial tuning of a given mouse with all trials stacked along axis=0
        Note that axes are permuted compared to tuning to avoid repeated transpose.
    label : numpy 1d array, shape (ntrial,)
        Context label of each trial.
    trial : numpy 1d array
        Indices of selected trials

    Returns
    -------
    X : numpy 2d array, shape (n_samples, ncell)
    y : numpy 1d array, shape (n_samples,) where n_samples = ntrial or ybin*ntrial
    '''
    ntrial, ybin, ncell = M.shape
    
    if trial is None:  # Take all trials
        trial = np.arange(ntrial)
    elif isinstance(trial, int):
        trial = [trial]
        
    # X = np.vstack([M[r] for r in trial])  # Stack ybin as samples
    # y = np.hstack([np.repeat(label[r], ybin) for r in trial])
    X = np.vstack([M[r].ravel() for r in trial])  # Stack ybin as features
    y = label[trial]
    
    return X, y

def shuffle_trial(M, label, seed=0):
    '''For each cell, shuffle the trial ID within each context.
    '''
    rng = np.random.default_rng(seed)
    Mshuffled = M.copy()
    for c in range(len(set(label))):  # nctx
        rs = np.where(label==c)[0]  # Trials within context
        for i in range(M.shape[2]):  # ncell
            Mshuffled[rs,:,i] = rng.permutation(M[rs,:,i])  # Shuffle by rows (trials)
            
    return Mshuffled

def shuffle_position(M, label, seed=0):
    '''For each trial, shuffle the position bins within each context
    '''
    rng = np.random.default_rng(seed)
    Mshuffled = M.copy()
    for r in range(M.shape[0]):  # ntrial
        Mshuffled[r,:,:] = rng.permutation(M[r,:,:])  # Shuffle by rows (y bins)
    
    return Mshuffled

def shuffle_label(label, seed=0):
    '''Shuffle the context labels.
    '''
    rng = np.random.default_rng(seed)
    return rng.permutation(label)

def decode_context_cv(M, label, nfold=10, shuffle=0):
    
    ntrial, ybin, ncell = M.shape
    accuracy = []
    coefficient = []
    if shuffle:
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
        
        ## Determine context by looking at the whole trial
        # nr = int(len(y_test)/ybin)
        # y_pred = np.reshape(y_pred, (nr,ybin))
        # y_test = np.reshape(y_test, (nr,ybin))
        # for r in range(nr):
        #     # accuracy.append(np.mean(y_test[r]==y_pred[r]))
        #     accuracy.append(get_majority(y_test[r], y_pred[r]))
        for r in range(len(y_test)):
            accuracy.append(y_pred[r]==y_test[r])
            
        # coefficient.append(np.ravel(clf.coef_))
        # np.column_stack(coefficient)
        coefficient.append(np.reshape(clf.coef_, (ybin, ncell)))
    
    return np.array(accuracy), np.stack(coefficient).mean(axis=0).T  # (ncell, ybin)

# %% Context decoder
def decode_context(datapath, day=1, min_rate=1/60, ybin=80, ysigma=1,
                   nfold=10, shuffle=0, seed=0):
    
    ## Load data
    data, cells, days, ctx, _ = get_data_bis(
        datapath, day=day, min_rate=min_rate, verbose=False)
    
    with h5py.File(datapath, 'r') as f:
        yrange = f['params/yrange'][()]  # Dataset dependent
    
    ## Prepare data (spatial tuning)
    tuning = tf.compute_tuning(data, yrange=yrange, ybin=ybin, ysigma=ysigma,
                               spike=False, transient=True)
    
    M = np.concatenate(tuning, axis=2)  # (ncell, ybin, ntrial)
    M = np.transpose(M, (2,1,0))  # (ntrial, ybin, ncell)
    label = np.hstack([np.repeat(ctx[k], tuning[k].shape[2])
                       for k in range(len(tuning))])  # (ntrial,)
    
    accuracy = [[] for _ in range(3)]  # original, shuffle position, shuffle context
    
    accuracy[0], coefficient = decode_context_cv(M, label, nfold=nfold, shuffle=shuffle)
    print('Mean accuracy (original) %.4g' % np.mean(accuracy[0]))
    
    ## Apply on shuffled data
    # M2 = shuffle_trial(M, label, seed=seed)
    M2 = shuffle_position(M, label, seed=seed)
    accuracy[1], _ = decode_context_cv(M2, label, nfold=nfold)
    
    label2 = shuffle_label(label, seed=seed)
    accuracy[2], _ = decode_context_cv(M, label2, nfold=nfold)
    
    return np.vstack(accuracy), coefficient

# %% Setup (for one experiment)
folder = r'D:\LW_Alldata\SI_Transient'
alldata = {'Enriched': glob(os.path.join(folder,'Enriched*.hdf5')),
           'Dissimilar': glob(os.path.join(folder,'Distinct*.hdf5')),
           'Similar': glob(os.path.join(folder,'Similar*.hdf5'))}
p = 2
exps = ['Enriched','Dissimilar','Similar']
datasets = alldata[exps[p]]
day = 1  # Recording days, None for all days
min_rate = 0/60  # 1/60  # Minimum transient rate for active cells
ybin = 20  # Number of spatial bins
ysigma = 0  # Size of the Gaussian filter (in number of spatial bin) for smoothing the tuning curves
shuffle = 2

# %% Check parameter (for one mouse)
datapath = datasets[0]
ybin_list = [4,6,8,10,16,20,30,40]
accuracy_list = np.zeros(len(ybin_list))

for i, ybin in enumerate(ybin_list):
    
    data, cells, days, ctx, _ = get_data_bis(
        datapath, day=day, min_rate=min_rate, verbose=False)
    
    with h5py.File(datapath, 'r') as f:
        yrange = f['params/yrange'][()]  # Dataset dependent
    
    ## Prepare data (spatial tuning)
    tuning = tf.compute_tuning(data, yrange=yrange, ybin=ybin, ysigma=ysigma,
                               spike=False, transient=True)
    
    M = np.concatenate(tuning, axis=2)  # (ncell, ybin, ntrial)
    M = np.transpose(M, (2,1,0))  # (ntrial, ybin, ncell)
    label = np.hstack([np.repeat(ctx[k], tuning[k].shape[2])
                       for k in range(len(tuning))])  # (ntrial,)
    
    accuracy, coefficient = decode_context_cv(M, label, nfold=10, shuffle=shuffle)
    accuracy_list[i] = np.mean(accuracy)
    
fig, ax = plt.subplots(figsize=(5,4))
ax.plot(ybin_list, accuracy_list)
fig.tight_layout()

# %% Run decoder
ybin = 20
shuffle = 2
accuracy_list = []
weight_list = []

for m, datapath in enumerate(datasets):
    
    print('Decoding mouse %d ...' % (m+1))
    
    accuracy, coefficient = decode_context(
        datapath, day=day, min_rate=min_rate, ybin=ybin, ysigma=ysigma, 
        nfold=10, shuffle=shuffle, seed=0)
    
    accuracy_list.append(accuracy)
    coeff_mean = np.mean(coefficient, axis=1)  # Average across spatial bin
    weight_list.append(coeff_mean/np.std(coeff_mean))  # Standardize each mouse

n_mice = len(accuracy_list)
df = ut.long_dataframe(accuracy_list, varnames=['Mouse','Data','Accuracy'],
                       varvalues=[np.arange(n_mice)+1,
                                  ['Original','Shuffle position','Shuffle context'],None])

fig, ax = plt.subplots(figsize=(6,5))
sns.barplot(df, x='Mouse', y='Accuracy', hue='Data', ax=ax)
ax.set(ylabel='Accuracy')
ax.legend(loc='best')
fig.tight_layout()

# %% Summary
def plot_summary(accuracy_list, color='C0'):
    
    summary = np.vstack([np.mean(accuracy, axis=1) for accuracy in accuracy_list])
    df_summary = pd.DataFrame(summary,
                              columns=['Original','Shuffle position','Shuffle context'])
    
    fig, ax = plt.subplots(figsize=(4,5))
    sns.barplot(df_summary, errorbar='se', width=0.6, capsize=0.3, linewidth=2, errwidth=2,
                errcolor=color, edgecolor=color, facecolor=(0,)*4)
    for point in summary:
        ax.plot(range(len(point)), point, c='gray', alpha=0.8, lw=1.2,
                marker='o', ms=6, mfc='none')
        
    pformat = {'pvalue_thresholds': [[1e-3,'***'],[0.01,'**'],[0.05,'*'],[1,'ns']]}
    pairs = [('Original','Shuffle position'),('Original','Shuffle context')]
    annot = Annotator(ax, pairs, data=df_summary)
    annot.configure(test='t-test_paired', loc='outside', pvalue_format=pformat)  # t-test_paired, Wilcoxon
    annot.apply_and_annotate()
    
    ax.set(ylabel='Accuracy')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    fig.tight_layout()

plot_summary(accuracy_list, color='tab:red')

# %% Save results
from PyQt5.QtWidgets import QFileDialog

def save_results(accuracy_list, weight_list):
    
    savename = QFileDialog.getSaveFileName(filter='XLSX (*.xlsx)')[0]
    n_mice = len(accuracy_list)
    df_list = []
    
    for accuracy in accuracy_list:
        df_list.append(pd.DataFrame(
            accuracy.astype(float).T,
            columns=['Original','Shuffle position','Shuffle context']))
        
    df = pd.DataFrame(np.hstack(weight_list), columns=['Weight'])
    
    with pd.ExcelWriter(savename) as writer:
        for m in range(n_mice):
            df_list[m].to_excel(writer, sheet_name=str(m+1), index=False)
        df.to_excel(writer, sheet_name='Weight', index=False)

save_results(accuracy_list, weight_list)



