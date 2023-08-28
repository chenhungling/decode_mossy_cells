# -*- coding: utf-8 -*-
"""
Summary of the running speed.
Related to Figure S1.

@author: Hung-Ling
"""
import os
from glob import glob
import numpy as np
import h5py
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator

os.chdir(r'C:\Users\Hung-Ling\Documents\Analysis_Python\MC_Analysis')
plt.style.use('figure.mplstyle')
import function.utils as ut

# %% Setup
folder = r'D:\LW_Alldata\SI_Transient'
alldata = {'Enriched': glob(os.path.join(folder,'Enriched*.hdf5')),
           'Dissimilar': glob(os.path.join(folder,'Distinct*.hdf5')),
           'Similar': glob(os.path.join(folder,'Similar*.hdf5'))}
p = 0
exps = ['Enriched','Dissimilar','Similar']
datapath = alldata[exps[p]][0]
day = 1  # Recording day (start from 1)
palettes = [['tab:gray','tab:red'],
            ['tab:gray','tab:blue'],
            ['tab:gray','tab:green']]

# %%
def get_behavior(datapath, day=1):
    
    data = dict()
    with h5py.File(datapath, 'r') as f:
        days = f.attrs['days']
        contexts = f.attrs['contexts']
        ctxnames = np.array(list(dict.fromkeys(contexts).keys()))  # ['Familiar','Novel'] Note that dict preserves order
        ctx = np.array([np.where(ctxnames==c)[0].item() for c in contexts])  # [0,1,0,1]
        data['fps'] = f.attrs['fps']
        data['yrange'] = f['params/yrange'][()]
        for key in ['t','y','v','moving']:
            data[key] = []
        ks = np.where(days==day)[0]
        for k in ks:
            g = f[str(k)]
            data['t'].append(g['n_frames'][()].astype(int))
            data['y'].append(g['y'][()])
            data['v'].append(g['speed'][()])
            data['moving'].append(g['moving'][()].astype(bool))
    
    return data, days[ks], ctx[ks]

data, days, ctx = get_behavior(datapath, day=day)

# %%
def speed_position(data, ctx=np.array([0,1,0,1]), c=0, ybin=80, vmax=60, plot=False):
    '''
    Calculate speed as a function of position.
    
    Returns
    -------
    V : 2D array, shape (ntrial, ybin)
    y : 1D array, shape (ybin,)
    '''
    yrange =  data['yrange']
    yedges = np.linspace(*yrange, ybin+1)
    y = (yedges[:-1] + yedges[1:])/2
    
    V = []
    for k in np.where(ctx==c)[0]:
        nframes = data['t'][k]
        nframes_ = np.hstack([0,np.cumsum(nframes)])
        ntrial = len(nframes)
        vs = np.zeros((ntrial, len(y)))
        for r in range(ntrial):
            seg = slice(nframes_[r], nframes_[r+1])
            moving = data['moving'][k][seg]
            yp = data['y'][k][seg][moving]
            vp = data['v'][k][seg][moving]
            ## Filter out bad y values at the begining and the end of the track
            mid = len(yp)//2
            t0 = np.argmin(yp[:mid:])
            t1 = mid + np.argmax(yp[mid::])
            yp = yp[t0:t1]
            vp = vp[t0:t1]
            ## Check that speed should be strictly positive and discard outside range position
            increasing = (vp > 0) & (yp>=yrange[0]) & (yp<=yrange[1])
            yp = yp[increasing]
            vp = vp[increasing]
            vs[r] = np.interp(y, yp, vp, left=np.NaN, right=np.NaN)
        vs[vs>vmax] = np.NaN  # Set max speed
        V.append(vs)
    V = np.vstack(V)
    
    if plot:
        fig, ax = plt.subplots()
        for v in V:
            ax.plot(y, v)
        fig.tight_layout()
        
    return V, y

V, y = speed_position(data, ctx=ctx, c=1, ybin=80, vmax=30, plot=True)

# %%
def summary_speed(datapaths, day=1, ybin=80, vmax=60, colors=['gray','C1'],
                  labels=['Fam','Nov'], plot=False):
    
    data_list = []
    for datapath in datapaths:
        data, days, ctx = get_behavior(datapath, day=day)
        data_list.append(data)
    
    nctx = len(set(ctx))
    Vlist = [[] for _ in range(nctx)]
    for c in range(nctx):
        for m, data in enumerate(data_list):
            V, _ = speed_position(data, ctx=ctx, c=c, ybin=ybin, vmax=vmax)
            # Vlist[c].append(V.mean(axis=0))  # Mean over trials (propagate NaN)
            # Vlist[c].append(np.nanmean(V, axis=0))  # Ignore NaN
            v = np.zeros(V.shape[1])
            for i, u in enumerate(V.T):
                if np.sum(np.isnan(u)) > u.size/2:  # An intermediate NaN policy
                    v[i] = np.NaN
                else:
                    v[i] = np.nanmean(u)
            Vlist[c].append(v)  # Collect data per mouse
        Vlist[c] = np.vstack(Vlist[c])  # Each row is one mouse
    
    yedges = np.linspace(0,4,ybin+1)  # Unify position y
    y = (yedges[:-1] + yedges[1:])/2
    
    if plot:
        fig, ax = plt.subplots(figsize=(4.2,3.6))
        for c in range(nctx):
            vmean = np.mean(Vlist[c], axis=0)
            verr = stats.sem(Vlist[c], axis=0)  # SEM over mice
            ax.fill_between(y, vmean-verr, vmean+verr, color=colors[c], alpha=0.2)
            ax.plot(y, vmean, color=colors[c], label=labels[c])
        ax.set(xlabel='Position (m)', ylabel='Speed (cm/s)', ylim=(5,30), xticks=[0,1,2,3,4])
        ax.legend(loc='lower right')
        fig.tight_layout()
    
    return Vlist

p = 2
datapaths = alldata[exps[p]]
palette = palettes[p]
Vlist = summary_speed(datapaths, day=1, ybin=80, vmax=40, colors=palette, plot=True)

for m in range(Vlist[0].shape[0]):
    for c in range(2):
        print('Mean running speed (context %d): %.4g' % (c, np.nanmean(Vlist[c][m])))

# %% Collect data of all experiment
day = 1
ybin = 8
vmax = 40
palettes = [['tab:gray','tab:red'],
            ['tab:gray','tab:blue'],
            ['tab:gray','tab:green']]
exps = ['Enriched','Dissimilar','Similar']
Vexp = []
for p in range(len(exps)):
    
    datapaths = alldata[exps[p]]
    Vlist = summary_speed(datapaths, day=day, ybin=ybin, vmax=vmax,
                          colors=palettes[p], labels=['Fam',exps[p]])
    Vexp.append([V.ravel() for V in Vlist])

# %% Compare all experiment
palette = ['tab:blue','tab:orange']
ff = 0.8  # Fill factor
width = ff/2  # Bar width

Vexp = ut.discard_nan(Vexp)
df = ut.long_dataframe(Vexp, varnames=['Experiment','Context','Speed'],
                       varvalues=[exps,['Fam','Nov'],None])

fig, ax = plt.subplots(figsize=(4.2,4.4))
for c in range(2):
    pos = np.arange(len(exps)) + (-ff/2 + width/2 + c*width)
    ax.boxplot([V[c] for V in Vexp], positions=pos, widths=width,
               showfliers=False, medianprops=dict(lw=2, c=palette[c]))
    
sns.swarmplot(data=df, x='Experiment', y='Speed', hue='Context', ax=ax,
              dodge=True, palette=palette)

pformat = {'pvalue_thresholds': [[1e-3,'***'],[1e-2,'**'],[0.05,'*'],[1,'ns']], 'fontsize':16}
pairs = [((p,'Fam'),(p,'Nov')) for p in exps]
annot = Annotator(ax, pairs, data=df, x='Experiment', y='Speed', hue='Context')
annot.configure(test='Wilcoxon', loc='outside', line_width=1.2, line_height=0., pvalue_format=pformat)
annot.apply_and_annotate()

ax.set(xlabel='', ylabel='Speed (cm/s)')
ax.set(xticks=np.arange(len(exps)), xlim=[-0.5, len(exps)-0.5])
ax.set_xticklabels(exps, rotation=20)
ax.legend(loc='best')
fig.tight_layout()

