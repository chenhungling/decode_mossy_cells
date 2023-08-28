# -*- coding: utf-8 -*-
"""
Get neuronal images from suite2p results and save the data into a hdf5 file.

@author: Hung-Ling
"""
import os
import h5py
import numpy as np
from scipy import sparse

# %%
def get_suite2p(folder, filepath=None, nplanes=1):
    
    if filepath is None:
        filepath = 'TempFile.hdf5'
        
    if nplanes == 1:
        stat = np.load(os.path.join(folder,'plane0','stat.npy'), allow_pickle=True)
        iscell = np.load(os.path.join(folder,'plane0','iscell.npy'))[:,0].astype(bool)
    else:
        stat = np.load(os.path.join(folder,'combined','stat.npy'), allow_pickle=True)
        iscell = np.load(os.path.join(folder,'combined','iscell.npy'))[:,0].astype(bool)
    
    cellind = np.where(iscell)[0]
    ncell = len(cellind)
    if nplanes == 1:
        iplane = np.zeros(ncell)
    else:
        iplane = np.array([stat[ind]['iplane'] for ind in cellind])
    
    with h5py.File(filepath, 'r+') as f:
        if 'suite2p' in f:
            del f['suite2p']
        g = f.create_group('suite2p')
        g.create_dataset('iplane', data=iplane)
    
    # %% Get neurons spatial footprint
    for p in range(nplanes):
        print('Getting data of imaging plane %d ...' % p)
        ops = np.load(os.path.join(folder,'plane'+str(p),'ops.npy'), allow_pickle=True).item()
        dims = (ops['Ly'], ops['Lx'])
        meanImg = ops['meanImg']
        meanImgE = ops['meanImgE']
        xrange, yrange = ops['xrange'], ops['yrange']
        maxImg = np.tile(np.NaN, dims).astype('float32')
        maxImg[slice(*yrange), slice(*xrange)] = ops['max_proj']
        corrImg = np.tile(np.NaN, dims).astype('float32')
        corrImg[slice(*yrange), slice(*xrange)] = ops['Vcorr']
        
        # nroi = np.sum(iplane==p)
        spatial = [] # csc_matrix((np.prod(dims), nroi), dtype=float)
        for i, ind in enumerate(cellind[iplane==p]):
            lam = stat[ind]['lam']
            ypix = np.mod(stat[ind]['ypix'], ops['Ly'])
            xpix = np.mod(stat[ind]['xpix'], ops['Lx'])
            roi = sparse.coo_matrix((lam/np.sum(lam), (ypix,xpix)), shape=dims)
            spatial.append(roi.reshape((-1,1), order='F').tocsc())  # Column-major order as in Caiman
        A = sparse.hstack(spatial)
        
        with h5py.File(filepath, 'r+') as f:
            g = f['suite2p'].create_group(str(p))
            g.create_dataset('meanImg', data=meanImg)
            g.create_dataset('meanImgE', data=meanImgE)
            g.create_dataset('maxImg', data=maxImg)
            g.create_dataset('corrImg', data=corrImg)
            h = g.create_group('A')
            h.create_dataset('data', data=A.data)
            h.create_dataset('indices', data=A.indices)
            h.create_dataset('indptr', data=A.indptr)
            h.attrs['shape'] = A.shape
            
# %%
if __name__ == '__main__':
    
    folder = r'X:\2-PhotonData\LWH\mc_morphing\5454\suite2p'
    filepath = r'D:\LW_mc_similar\Data mice\Similar_8_mouse_5454_day3.hdf5'
    get_suite2p(folder, filepath=filepath, nplanes=3)
