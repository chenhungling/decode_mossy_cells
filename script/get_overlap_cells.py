# -*- coding: utf-8 -*-
"""
Compute pairwise distance and correlation from invivo dataset to identify possible repeated cells.
Remove repeated cells from the suite2p combined folder (manually define cells to be deleted).

@author: Hung-Ling
"""
import os
from glob import glob
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from scipy.ndimage.measurements import center_of_mass
from scipy.spatial.distance import pdist, squareform

os.chdir(r'C:\Users\Hung-Ling\Documents\Analysis_Python\MC_Analysis')
from function.data2p import get_data_bis
import function.tuning_function as tf

# %%
def get_distances(datapath, zpix=10):
    
    with h5py.File(datapath, 'r') as f:
        iplane = f['suite2p/iplane'][()]
        nplanes = len(set(iplane))
        dims = f['suite2p/0/meanImg'][()].shape
        Ap = []  # Spatial components for each plane
        for p in range(nplanes):
            g = f['suite2p/'+str(p)]
            Ap.append(
                csc_matrix((g['A/data'][()], g['A/indices'][()], g['A/indptr'][()]),
                           shape=g['A'].attrs['shape']))
    img_components = []
    for A in Ap:
        img_components.extend(
            [A[:,i].reshape(dims, order='F').toarray()
             for i in range(A.shape[1])])
    cms = np.array([center_of_mass(comp) for comp in img_components])  # Shape (K,2) centroid (y,x) of each component
    cms_3d = np.hstack([cms, iplane.reshape((-1,1))*zpix])
    dist = squareform(pdist(cms_3d, metric='euclidean'))
    
    return dist

def show_pair(M, pair, selected_cells=None):
    
    i, j = pair
    if selected_cells is None:
        selected_cells = np.arange(M.shape[0])
    
    fig, axs = plt.subplots(1,2,sharex=True,sharey=True,figsize=(4,3))
    axs[0].imshow(M[i].T, interpolation='none')
    axs[0].set_title('Cell %d' % selected_cells[i])
    axs[1].imshow(M[j].T, interpolation='none')
    axs[1].set_title('Cell %d' % selected_cells[j])
    for ax in axs:
        ax.set_aspect('auto')
    fig.tight_layout()
    
# %% Setup
folder = r'D:\LW_Alldata\SI_Transient'
alldata = {'Enriched': glob(os.path.join(folder,'Enriched*.hdf5')),
           'Dissimilar': glob(os.path.join(folder,'Distinct*.hdf5')),
           'Similar': glob(os.path.join(folder,'Similar*.hdf5'))}
p = 0
exps = ['Enriched','Dissimilar','Similar']
datapath = alldata[exps[p]][2]
day = 1  # Recording days, None for all days
ybin = 40  # Number of spatial bins
ysigma = 1  # Size of the Gaussian filter (in number of spatial bin) for smoothing the tuning curves

# %% Load all cells
data, cells, days, ctx, selected_cells = get_data_bis(datapath, day=day, min_rate=0)

with h5py.File(datapath, 'r') as f:
    yrange = f['params/yrange'][()]  # Dataset dependent

ncell = data['F'][0].shape[0]
print('Recording days:', days)
print('Context fam/nov:', ctx)
print('Number of cells:', ncell)

## Spatial tuning
tuning = tf.compute_tuning(data, yrange=yrange, ybin=ybin, ysigma=ysigma,
                           spike=False, transient=True)
M = []
for c in range(2):
    ks = np.where(ctx==c)[0]
    M.append(np.concatenate([tuning[k].squeeze() for k in ks], axis=2))  # (ncell, ybin, ntrial)
M = np.concatenate(M, axis=2)

## Compute pairwise correlation
R = np.corrcoef(M.reshape((ncell,-1)))
rs = R[np.triu_indices(ncell, k=1)]

# %% Compute pairwise distance
D = get_distances(datapath, zpix=10)

assert D.shape[0] == ncell, 'Number of cells mismatched !!!'

ds = D[np.triu_indices(ncell, k=1)]

# %% Scatter plot
candidate = ds<50

fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(ds[candidate], rs[candidate])
ax.set(xlabel='Distance (pixel)', ylabel='Correlation')

# %%
corr_thr = 0.4
dist_thr = 50

for i in range(ncell):
    for j in range(i+1,ncell):
        if R[i,j] > corr_thr and D[i,j] < dist_thr:
            show_pair(M, (i,j))
                        
# %% Remove cells from the suite2p combined folder
suite2p_folder = r'D:\LW_mc_similar\8_5454\suite2p'
iscell_file = os.path.join(suite2p_folder,'combined','iscell.npy')
iscell = np.load(iscell_file)
print('Original number of cells:', int(np.sum(iscell[:,0])))

delete_cells = [15,57]
cellid = np.where(iscell[:,0])[0]
# assert len(cellid) == ncell, 'Number of cells mismatched !!!'

# %%
for i in delete_cells:
    iscell[cellid[i],0] = 0

print('New number of cells:', int(np.sum(iscell[:,0])))
np.save(iscell_file, iscell)


