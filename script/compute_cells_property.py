# -*- coding: utf-8 -*-
"""
Take invivo data file (see suite2p_workflow.py) and compute cells properties
in each session (category):
    - Mean activity: transient/spike and auc rate
    - Spatial information (original and shuffled with multiprocessing)
    - Run-to-run reliability (tuning stability)

Note: execute this script in an external system terminal for multiprocessing
    
@author: Hung-Ling
"""
import os
import numpy as np
import h5py
from scipy.ndimage import gaussian_filter1d
import multiprocessing as mp
from tqdm import tqdm

os.chdir(r'C:\Users\Hung-Ling\Documents\Analysis_Python\MC_Analysis')
from function.transient import transient_rate
import function.information as info
from function.tuning_function import tuning_1d

# %%
def main(datapath):
    
    yrange, ybin = (0.05,4.05), 80  # 0.05 m per spatial bin
    # with h5py.File(datapath, 'r') as f:  # Get previously used parameters
    #     yrange = f['params/yrange'][()]
    #     ybin = int(f['params/ybin'][()])
    ysigma = 1  # Size of the Gaussian filter (in number of spatial bin) for smoothing the spatial tuning curves
    spike = False  # Whether to use spike data
    shuffle = 'rotate'  # 'rotate'|'segment'
    window = 50  # Break y into segments of {window} time points then shuffle (shuffle='segment')
    n_shuffle = 1000  # Number of boostrapping spatial information
    random_state = 0
    n_processes = 4
        
    # %% Read simple data file
    data, days, ctx = get_data(datapath)
    ncat = len(data['F'])
    ncells = data['F'][0].shape[0]  # [data['F'][m][0].shape[0] for m in range(n_mice)]
    trate =  np.zeros((ncells, ncat))  # [np.zeros((nc, nday*nctx)) for nc in ncells]
    tauc =  np.zeros((ncells, ncat))  # [np.zeros((nc, nday*nctx)) for nc in ncells]
    srate = np.zeros((ncells, ncat))  # [np.zeros((nc, nday*nctx)) for nc in ncells]
    sauc = np.zeros((ncells, ncat))  # [np.zeros((nc, nday*nctx)) for nc in ncells]
    si = np.zeros((ncells, ncat))  # [np.zeros((nc, nday*nctx)) for nc in ncells]
    reliability = np.zeros((ncells, ncat))
    
    # %% Spatial tuning
    tuning = []
    for k in range(ncat):
        ntrial = len(data['t'][k])
        nframes = np.hstack([0, np.cumsum(data['t'][k])])
        dFoY = np.zeros((ncells, ybin, ntrial))
        for r in range(ntrial):
            seg = slice(nframes[r], nframes[r+1])
            ytmp = data['y'][k][seg]
            movetmp = data['moving'][k][seg]
            Ftmp = data['F'][k][:,seg]*data['Tr'][k][:,seg]
            dFoY[:,:,r] = tuning_1d(Ftmp[:,movetmp], ytmp[movetmp],
                                    yrange=yrange, ybin=ybin)[1]
        dFoY[np.isnan(dFoY)] = 0  # NaN due to unoccupied y position
        dFoY[dFoY < 0] = 0  # Discard unusual transient giving rise to negative fluor signal
        if ysigma > 0:
            dFoY = gaussian_filter1d(dFoY, ysigma, axis=1, mode='nearest')
        tuning.append(dFoY)
        
    # %% Mean activities (cell-by-cell) and spatial information
    for k in range(ncat):
        y = data['y'][k]
        yin = (y>=yrange[0]) & (y<yrange[1])  # Avoid activity outside yrange increases SI after shuffling position vector... 
        moving = data['moving'][k] & yin
        trace = data['F'][k][:,moving]
        transient = data['Tr'][k][:,moving]
        spiking = data['Sp'][k][:,moving]
        y = y[moving]
        
        ## Single cells properties
        for i, (a, b) in enumerate(zip(trace, transient)):
            _, rate_, auc_ = transient_rate(a, b, fps=data['fps'])
            trate[i,k] = rate_
            tauc[i,k] = auc_
        for i, spks in enumerate(spiking):  # (ncell, T)
            srate[i,k] = np.sum(spks > 0)/(len(spks)/data['fps'])
            sauc[i,k] = np.sum(spks)/(len(spks)/data['fps'])
        
        F = spiking if spike else trace*transient
        si[:,k] = info.spatial_information_1d(F, y, yrange=yrange, ybin=ybin,
                                              fps=data['fps'], spike=True)
        ## Run-to-run reliability
        for i in range(ncells):
            M = tuning[k][i]  # (ybin, ntrial)
            bad_trial = np.where(M.var(axis=0)==0)[0]
            if len(bad_trial) > 0:
                M = np.delete(M, bad_trial, axis=1)
            ntrial = M.shape[1]
            if ntrial > 1:
                R = np.corrcoef(M.T)[np.triu_indices(ntrial, k=1)]  # All trial-pairs (upper triangular part)
                # R = np.diag(np.corrcoef(M.T), k=1)  # Adjacent trials (off diagonal part)
                reliability[i,k] = np.mean(R)
            else:
                reliability[i,k] = np.NaN
                    
    with h5py.File(datapath, 'r+') as f:
        for key in ['transient_rate','transient_auc','spike_rate','spike_auc','spatial_info','reliability']:
            if key in f:
                del f[key]
        f.create_dataset('transient_rate', data=trate)
        f.create_dataset('transient_auc', data=tauc)
        f.create_dataset('spike_rate', data=srate)
        f.create_dataset('spike_auc', data=sauc)
        f.create_dataset('spatial_info', data=si)
        f.create_dataset('reliability', data=reliability)
        
    # %% Boostrapping
    si_shuffle = np.zeros((n_shuffle, ncells, ncat))
    for k in range(ncat):
        
        position = data['y'][k]
        yin = (position>=yrange[0]) & (position<yrange[1])  # Avoid activity outside yrange increases SI after shuffling position vector...
        moving = data['moving'][k] & yin
        trace = data['F'][k][:,moving]
        transient = data['Tr'][k][:,moving]
        spiking = data['Sp'][k][:,moving]
        F = spiking if spike else trace*transient
        
        ntrial = len(data['t'][k])
        nframes = np.hstack([0, np.cumsum(data['t'][k])])
        ys = []
        for r in range(ntrial):
            seg = slice(nframes[r],nframes[r+1])
            ys.append(position[seg][moving[seg]])
        
        rng = np.random.default_rng(random_state)  # Fix Random Number Generator for reproducibility
        params = []
        if shuffle == 'rotate':
            for s in range(n_shuffle):
                yshuffle = []
                for y in ys:  # Rotate for each trial
                    t = rng.choice(len(y))
                    yshuffle.append(np.hstack([y[t:], y[:t]]))
                params.append((F, np.hstack(yshuffle), yrange, ybin, data['fps'], True))
        elif shuffle == 'segment':
            y = np.hstack(ys)
            T = len(y)
            nsegment = T//window
            t_split = np.array_split(np.arange(T), nsegment)
            for s in range(n_shuffle):
                randorder = rng.permutation(nsegment)
                y_shuffle = np.hstack([y[t_split[r]] for r in randorder])
                params.append((F, y_shuffle, yrange, ybin, data['fps'], True))
            
        print(f'Computing for category {k} ...')
        with mp.Pool(processes=n_processes) as pool:    
            res = tqdm(pool.imap(si1d_wrapper, params), total=n_shuffle)
            si_shuffle[:,:,k] = np.vstack(list(res))
    
    with h5py.File(datapath, 'r+') as f:
        if 'spatial_info_shuffle' in f:
            del f['spatial_info_shuffle']
        f.create_dataset('spatial_info_shuffle', data=si_shuffle)
        for key in ['yrange','ybin','ysigma','spike','shuffle','window','n_shuffle','random_state']:
            if key in f['params']:
                del f['params/'+key]
        g = f['params']
        g['yrange'] = yrange
        g['ybin'] = ybin
        g['ysigma'] = ysigma
        g['spike'] = spike
        g['shuffle'] = shuffle
        g['window'] = window
        g['n_shuffle'] = n_shuffle
        g['random_state'] = random_state
    
# %%
def get_data(datapath):
    data = dict()
    with h5py.File(datapath, 'r') as f:
        ncat = f.attrs['n_categories']
        days = f.attrs['days']
        contexts = f.attrs['contexts']
        ctxnames = np.array(list(dict.fromkeys(contexts).keys()))  # ['Familiar','Novel'] Note that dict preserves order
        ctx = np.array([np.where(ctxnames==c)[0].item() for c in contexts])  # [0,1,0,1]
        data['fps'] = f.attrs['fps']
        for key in ['t','y','moving','F','Tr','Sp']:
            data[key] = []
        for k in range(ncat):
            g = f[str(k)]
            data['t'].append(g['n_frames'][()].astype(int))
            data['y'].append(g['y'][()])
            data['moving'].append(g['moving'][()].astype(bool))
            data['F'].append(g['F'][()])
            data['Tr'].append(g['transient'][()].astype(bool))
            data['Sp'].append(g['spike'][()])
    return data, days, ctx

# %%
def si1d_wrapper(params):
    F, y, yrange, ybin, fps, spike = params
    si = info.spatial_information_1d(F, y, yrange=yrange, ybin=ybin, fps=fps, spike=spike)
    return si
        
# %%
if __name__ == '__main__':
    
    # datapath = r'D:\LW_Alldata\SI_Transient\Enriched3_6160_invivo.hdf5'
    # main(datapath)
    
    # from glob import glob
    # datapaths = glob(os.path.join(r'D:\LW_alldata_spike','*.hdf5'))
    datapaths = [
        # r'D:\LW_Alldata\SI_Transient\Enriched1_6044_invivo.hdf5',
        # r'D:\LW_Alldata\SI_Transient\Enriched2_6045_invivo.hdf5',
        # r'D:\LW_Alldata\SI_Transient\Enriched3_6160_invivo.hdf5',
        # r'D:\LW_Alldata\SI_Transient\Enriched4_6161_invivo.hdf5']
        # r'D:\LW_Alldata\SI_Transient\Distinct1_111_invivo.hdf5',
        # r'D:\LW_Alldata\SI_Transient\Distinct2_114_invivo.hdf5',
        # r'D:\LW_Alldata\SI_Transient\Distinct3_132_invivo.hdf5']
        r'D:\LW_Alldata\SI_Transient\Similar1_182_invivo.hdf5',
        r'D:\LW_Alldata\SI_Transient\Similar2_1506_invivo.hdf5',
        r'D:\LW_Alldata\SI_Transient\Similar3_6815_invivo.hdf5',
        r'D:\LW_Alldata\SI_Transient\Similar4_6826_invivo.hdf5',
        r'D:\LW_Alldata\SI_Transient\Similar5_937_invivo.hdf5',
        r'D:\LW_Alldata\SI_Transient\Similar6_939_invivo.hdf5',
        r'D:\LW_Alldata\SI_Transient\Similar7_948_invivo.hdf5',
        r'D:\LW_Alldata\SI_Transient\Similar8_5454_invivo.hdf5']
    for datapath in datapaths:
        main(datapath)
    
    
    
    