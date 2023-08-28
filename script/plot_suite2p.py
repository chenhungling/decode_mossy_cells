# -*- coding: utf-8 -*-
"""
Example plot for fluorescence treatment with subtraction of red channel signal.
Related to Figure S2.

@author: Hung-Ling
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

os.chdir(r'C:\Users\Hung-Ling\Documents\Analysis_Python\MC_Analysis')
plt.style.use('figure.mplstyle')

from function.data2p import get_data_bis

# %%
def subtract_channel(x, y, bad_frames=None, percentile=75, plot=False):
    
    if bad_frames is not None:  # Exclude outlier frames
        x2 = np.copy(x)[~bad_frames]
        y2 = np.copy(y)[~bad_frames]
    else:
        x2 = np.copy(x)
        y2 = np.copy(y)
    ix = x2<np.percentile(x2, percentile)  # Avoid taking saturated red-channel values
    iy = y2<np.percentile(y2, percentile)  # Avoid taking visible calcium transients
    idx = np.logical_and(ix, iy)
    if np.sum(idx) > 32:
        a, b = np.polyfit(x[idx], y[idx], 1)  # Linear regression
        background = a*x
        ysub = y - background + np.mean(background)  # Subtract linearly scaled x from y, without changing mean of y
    else:
        background = np.zeros_like(x)
        ysub = y
        print('Warning: too few good frames, recommend lowering corr_thr, or increase prct_thr')
    if plot:
        fig, ax = plt.subplots(figsize=(4.6,4.5))
        ax.plot(x, y, ls='none', marker='o', ms=3, c='gray')
        ax.plot(x2[idx], y2[idx], ls='none', marker='o', ms=3.2, c='tab:orange')
        xline = np.linspace(x.min(), x.max(), 100)
        yline = a*xline+b
        ax.plot(xline, yline, c='tab:red', ls='--', lw=2)
        ax.set_xlabel('Red channel intensity (a.u.)')
        ax.set_ylabel('Green channel intensity (a.u.)')
        fig.tight_layout()
        
    return ysub, background

# %% Load data
suite2p_folder = r'D:\LW_mc_enriched\6160\suite2p'
nplane = 3

iscell = np.load(os.path.join(suite2p_folder,'combined','iscell.npy'))[:,0].astype(bool)
ops = np.load(os.path.join(suite2p_folder,'combined','ops.npy'), allow_pickle=True).item()
fps = ops['fs']
F, F2 = [], []
for p in range(nplane):
    data_folder = os.path.join(suite2p_folder,'plane'+str(p))
    F.append(np.load(os.path.join(data_folder,'F.npy')))
    F2.append(np.load(os.path.join(data_folder,'F_chan2.npy')))
F = np.vstack(F)
F2 = np.vstack(F2)

nframes = ops['nframes_per_file'].astype(int)
nframes_ = np.hstack([0, np.cumsum(nframes)])
nfile = len(nframes)
nroi = F.shape[0]
ncell = np.sum(iscell)
cellid = np.where(iscell)[0]

fig, ax = plt.subplots(figsize=(12,6))
img = ax.imshow(F[cellid], cmap='gray_r', interpolation='none')
ax.set_aspect('auto')
fig.colorbar(img, ax=ax)
fig.tight_layout()

# %% Check baseline fluctuation
pca = PCA(n_components=20)
X = pca.fit_transform(F2[cellid].T)

i0 = np.argmax(pca.components_[0])
i1 = np.argmax(pca.components_[1])
print('Cell %d is the most similar to 1st PC' % i0)
print('Cell %d is the most similar to 2nd PC' % i1)

fig, axs = plt.subplots(3,1,sharex=True,figsize=(12,8))
axs[0].imshow(X.T, cmap='gray_r', interpolation='none')
axs[0].set_aspect('auto')
axs[1].plot(X.T[0], c='C0', lw=0.5)  # F2[cellid[i0]]
axs[2].plot(X.T[1], c='C1', lw=0.5)  # F2[cellid[i1]]
fig.tight_layout()

# %% Example trace with scatter plot
i = 17
j = 4
prct = 90

seg = np.arange(nframes_[j], nframes_[j+1])
t = seg/fps
trace1 = F[cellid[i], seg]  # Green channel
trace2 = F2[cellid[i], seg]  # Red channel

trace_sub, background = subtract_channel(
    trace2, trace1, bad_frames=None, percentile=prct, plot=True)  # Attention NOT to inverse order trace2, trace1

fig, axs = plt.subplots(3,1,sharex=True,figsize=(6.5,4.5))
axs[0].plot(t, trace1, c='tab:green', label='Green channel')
axs[1].plot(t, trace2, c='tab:red', label='Red channel')
axs[2].plot(t, trace_sub, c='tab:blue', label='Corrected trace')
axs[1].set(ylabel='Intensity (a.u.)')
axs[2].set(xlabel='Time (s)')
for ax in axs:
    ax.legend()
fig.tight_layout()
fig.subplots_adjust(hspace=0.1)

# %% Example map
trials = np.arange(8)
ts = np.hstack([np.arange(nframes_[r], nframes_[r+1]) for r in trials])
T = len(ts)

Green = F[cellid][:,ts]
Red = F2[cellid][:,ts]
# order = np.argsort(np.mean(Red, axis=1))[::-1]
order = np.arange(len(cellid))

fig, axs = plt.subplots(2,1,sharex=True,figsize=(10,6))
img0 = axs[0].imshow(Green[order], cmap='gray_r', interpolation='none',
                     vmin=0, vmax=20000, extent=[0,T/fps,ncell,0])
img1 = axs[1].imshow(Red[order], cmap='gray_r', interpolation='none',
                     vmin=0, vmax=30000, extent=[0,T/fps,ncell,0])
for r in trials[1::]:
    axs[0].axvline(nframes_[r]/fps, c='tab:green', ls='--', lw=1.5)
    axs[1].axvline(nframes_[r]/fps, c='tab:red', ls='--', lw=1.5)
for ax in axs:
    ax.set_aspect('auto')
    ax.set_ylabel('Cell')
axs[1].set_xlabel('Time (s)')
fig.colorbar(img0, ax=axs[0])
fig.colorbar(img1, ax=axs[1])
fig.tight_layout()
fig.subplots_adjust(hspace=0.1)

# %% Plot example calcium transients
datapath = r'D:\LW_Alldata\SI_Transient\Enriched3_6160_invivo.hdf5'
day = 1  # Recording days, None for all days
min_rate = 1/60  # Minimum transient rate for active cells

## Load data
data, cells, days, ctx, selected_cells = get_data_bis(
    datapath, day=day, min_rate=min_rate, verbose=True)
    
print('Context fam/nov:', ctx)
print('Active cells:', selected_cells)

# %%
cellid = [np.where(selected_cells==i)[0].item() for i in [17,18,33,34,36,41,47]]  # cellid: indices in active subset of cells
trialid = [range(0,8),[],[],[]]
colors = [plt.cm.tab10(j) for j in [0,1,2,3,8,6,9]]
fps = data['fps']
t0 = 0  # Time offset

fig, ax = plt.subplots(figsize=(11,5))

for k in range(len(trialid)):
    if len(trialid[k]) > 0:
        nframes = data['t'][k][trialid[k]]
        nframes_ = np.hstack([0, np.cumsum(nframes)])
        sub = np.hstack([np.arange(nframes_[r], nframes_[r+1]) for r in trialid[k]])
        t = t0 + np.arange(len(sub))/fps
        F = data['F'][k][:,sub]
        Tr = data['Tr'][k][:,sub]
        moving = data['moving'][k][sub]
        for i, ii in enumerate(cellid[::-1]):
            ax.plot(t, 2*i+F[ii], lw=1, c='gray', alpha=0.8)      
            F2 = F[ii].copy()
            F2[~Tr[ii]] = np.NaN
            ax.plot(t, 2*i+F2, lw=1.5, c=colors[::-1][i])
        t0 += len(sub)/fps
        for r in nframes_[1:-1:]:
            ax.axvline(r/fps, c='gray', ls='--', lw=1.5)
ax.set(yticks=range(0,2*len(cellid),2), yticklabels=selected_cells[cellid][::-1],
       ylabel='Cell', xlabel='Time (s)', xlim=[0,t0])
fig.tight_layout()