# -*- coding: utf-8 -*-
"""
After running: 
    - Suite2p with iscell check
    - FluorSpikes (post treatment of fluorescence traces, spike deconvolution
                   and significant transient)
This script collects two-photon data from Suite2p and FluorSpikes outcomes,
behavior data (ephys files), and then create a hdf5 file storing jointly the
invivo data of one mouse. Data are organized according to user specified category
in an excel table ('Files_*.xlsx')

@author: Hung-Ling
"""
import os
from glob import glob
import numpy as np
import h5py
import pandas as pd
from scipy.interpolate import interp1d

os.chdir(r'C:\Users\Hung-Ling\Documents\Analysis_Python\MC_Analysis')
import function.behavior as beh
from get_suite2p_data import get_suite2p

# %% Setup data path

data_folder = r'Y:\2-PhotonData\LWH\mc_morphing\5454\Rerun'
suite2p_folder = r'D:\LW_mc_similar\8_5454\suite2p'
savepath = r'D:\LW_Alldata\SI_Transient\Similar8_5454_invivo.hdf5'

onset = 1  # Discard the 1st ephys frame (as 1st sbx frame was dropped)
yspan = 5  # Size for smoothing position

# %%
def main(data_folder, suite2p_folder, savepath):
        
    ## Load suite2p data
    print('Get suite2p data ...')
    plane_folders = sorted([f.path for f in os.scandir(suite2p_folder) 
                            if f.is_dir() and f.name[:5]=='plane'])
    nplanes = len(plane_folders)
    if nplanes == 1:
        opsfile = os.path.join(suite2p_folder,'plane0','ops.npy')
    else:
        opsfile = os.path.join(suite2p_folder,'combined','ops.npy')
    ops = np.load(opsfile, allow_pickle=True).item()
    
    fps = ops['fs']
    T = ops['nframes']
    if 'nframes_per_file' in ops.keys():  # Corrected io/sbx.py by HLC
        nframes = ops['nframes_per_file']
    elif 'nframes_per_folder' in ops.keys():  # Old io/sbx.py
        nframes = (ops['nframes_per_folder']/ops['nchannels']).astype(int)
    else:
        nframes = np.array([ops['nframes']])
    nframes_ = np.hstack([0, np.cumsum(nframes)])
    assert T == nframes_[-1]
    
    # %% Load curated two-photon data (fluorspikes)
    hdf5file = glob(os.path.join(suite2p_folder, '*fluorspikes.hdf5'))[0]
    
    print('Get fluorspikes data ...')
    with h5py.File(hdf5file, 'r') as f:
        g = f['/fluorspikes']
        F = g['F'][()]
        spike = g['S'][()]
        transient = g['G'][()]
    
    ncells = F.shape[0]
        
    # %% Load behavior data and organize into categories
    excelfile = glob(os.path.join(data_folder, 'Files_*.xlsx'))[0]
    A = pd.read_excel(excelfile)
    
    count = 0
    ks, contexts, days = [], [], []
    categories = np.zeros(A.shape[0], dtype=int)
    for j, ctx in enumerate(A['Context']):
        if len(ks) == 0 or ctx != contexts[-1]:
            ks.append(count)
            contexts.append(ctx)
            days.append(A['Day'][j])
            categories[j:] = count
            count += 1
            
    ncat = len(ks)
    tlist = [[] for _ in range(ncat)]
    ylist = [[] for _ in range(ncat)]
    vlist = [[] for _ in range(ncat)]
    movelist = [[] for _ in range(ncat)]
    Flist = [[] for _ in range(ncat)]
    Slist = [[] for _ in range(ncat)]
    Trlist = [[] for _ in range(ncat)]
    
    print('Get behavior data ...')
    for j, fname in enumerate(A['Filename']):
        
        print('Processing file %d/%d' % (j+1, A.shape[0]))
        k = categories[j]
        ephysfile = os.path.join(data_folder, fname+'.ephys')
        
        offset = int(nplanes*nframes[j]) + 1
        y = beh.get_behavior_data(ephysfile, onset=onset, offset=offset)
        y = np.nanmean(y.reshape((-1, nplanes)), axis=1)
        
        missing = np.isnan(y)
        if np.any(missing):
            t0 = np.arange(len(y))[~missing]
            t1 = np.arange(len(y))[missing]
            f = interp1d(t0, y[~missing], kind='nearest', assume_sorted=True)
            y[missing] = f(t1)
            
        ysmooth = beh.smooth_position(y, span=yspan)
        v, ismoving = beh.moving(ysmooth, minspeed=5, fps=fps, y2cm=100)
        
        tlist[k].append(nframes[j])
        ylist[k].append(ysmooth)
        vlist[k].append(v)
        movelist[k].append(ismoving)
        
        seg = slice(nframes_[j],nframes_[j+1])
        Flist[k].append(F[:,seg])
        Slist[k].append(spike[:,seg])
        Trlist[k].append(transient[:,seg])
        
    # %% Save hdf5
    print('Creating a hdf5 file ...')
    with h5py.File(savepath, 'a') as f:
        f.attrs['n_categories'] = ncat
        f.attrs['n_cells'] = ncells
        f.attrs['contexts'] = contexts
        f.attrs['days'] = days
        f.attrs['fps'] = fps
        for k in range(ncat):
            g = f.create_group(str(k))
            g['n_frames'] = tlist[k]
            g['y'] = np.hstack(ylist[k])
            g['speed'] = np.hstack(vlist[k])
            g['moving'] = np.hstack(movelist[k])
            g['F'] = np.hstack(Flist[k])
            g['spike'] = np.hstack(Slist[k])
            g['transient'] = np.hstack(Trlist[k])
        g = f.create_group('params')
        g['yspan'] = yspan
    
    ## Mount additional suite2p data (spatial footprint)
    get_suite2p(suite2p_folder, filepath=savepath, nplanes=nplanes)
    print('Saved:', savepath)

# %%
if __name__ == '__main__':

    main(data_folder, suite2p_folder, savepath)
    
    
      