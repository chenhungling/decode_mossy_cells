# -*- coding: utf-8 -*-
"""
Take invivo data file (see suite2p_workflow.py) and compute cells properties
in each context (per day):
    - Mean activity: transient/spike and auc rate
    - Spatial information (original and shuffled with multiprocessing)
    - Run-to-run reliability (tuning stability)
    - Context selectivity score (Novel - Familiar)
    - Remapping index (correlation FF - FN)

Note: execute this script in an external system terminal for multiprocessing
    
@author: Hung-Ling
"""
import os
import numpy as np
import h5py
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import multiprocessing as mp
from tqdm import tqdm

os.chdir(r'C:\Users\Hung-Ling\Documents\Analysis_Python\MC_Analysis')
from function.transient import transient_rate
import function.information as info
from function.tuning_function import tuning_1d

# %%
def main(datapath):
    
    n_processes = 4
    # yrange, ybin = (0.15,4.05), 80  # 0.05 m per spatial bin
    # ysigma = 1  # Size of the Gaussian filter (in number of spatial bin) for smoothing the spatial tuning curves
    # spike = False  # Whether to use spike data
    # shuffle = 'rotate'  # 'rotate'|'segment'
    # window = 50  # Break y into segments of {window} time points then shuffle (shuffle='segment')
    # n_shuffle = 1000  # Number of boostrapping spatial information
    # random_state = 0
    
    # %% Get previously used parameters
    with h5py.File(datapath, 'r') as f:
        g = f['params']
        yrange = g['yrange'][()]
        ybin = int(g['ybin'][()])
        ysigma = float(g['ysigma'][()])
        spike = bool(g['spike'][()])
        shuffle = g['shuffle'][()]
        window = int(g['window'][()])
        n_shuffle = int(g['n_shuffle'][()])
        random_state = int(g['random_state'][()])

    # %% Read simple data file
    data, days, ctx = get_data(datapath)
    
    nday = len(set(days))
    nctx = len(set(ctx))
    ncells = data['F'][0].shape[0]
    trate =  np.zeros((ncells, nday*nctx))
    tauc =  np.zeros((ncells, nday*nctx))
    srate = np.zeros((ncells, nday*nctx))
    sauc = np.zeros((ncells, nday*nctx))
    si = np.zeros((ncells, nday*nctx))
    reliability = np.zeros((ncells, nday*nctx))
    tuning = []
    
    # %% Mean activities (cell-by-cell) and spatial information
    count = 0
    
    for d in range(nday):
        for c in range(nctx):
            
            ks = np.where((days==(d+1)) & (ctx==c))[0]  # Day starts from 1
            y = np.hstack([data['y'][k] for k in ks])
            yin = (y>=yrange[0]) & (y<yrange[1])  # Avoid activity outside yrange increases SI after shuffling position vector... 
            moving = np.hstack([data['moving'][k] for k in ks]) & yin
            trace = np.hstack([data['F'][k] for k in ks])
            transient = np.hstack([data['Tr'][k] for k in ks])
            spiking = np.hstack([data['Sp'][k] for k in ks])
            
            ## Single cells properties
            for i, (a, b) in enumerate(zip(trace[:,moving], transient[:,moving])):
                _, rate_, auc_ = transient_rate(a, b, fps=data['fps'])
                trate[i,count] = rate_
                tauc[i,count] = auc_
            for i, spks in enumerate(spiking):  # (ncell, T)
                srate[i,count] = np.sum(spks > 0)/(len(spks)/data['fps'])
                sauc[i,count] = np.sum(spks)/(len(spks)/data['fps'])
            
            F = spiking[:,moving] if spike else trace[:,moving]*transient[:,moving]
            si[:,count] = info.spatial_information_1d(
                F, y[moving], yrange=yrange, ybin=ybin, fps=data['fps'], spike=True)
            
            ## Spatial tuning of each trial
            nframes = np.hstack([data['t'][k] for k in ks])
            nframes_ = np.hstack([0, np.cumsum(nframes)])
            ntrial = len(nframes)
            dFoY = np.zeros((ncells, ybin, ntrial))
            for r in range(ntrial):
                seg = slice(nframes_[r], nframes_[r+1])
                ytmp = y[seg]
                movetmp = moving[seg]
                if spike:
                    Ftmp = spiking[:,seg]
                else:
                    Ftmp = trace[:,seg]*transient[:,seg]
                dFoY[:,:,r] = tuning_1d(Ftmp[:,movetmp], ytmp[movetmp],
                                        yrange=yrange, ybin=ybin)[1]
            dFoY[np.isnan(dFoY)] = 0  # NaN due to unoccupied y position
            dFoY[dFoY < 0] = 0  # Discard unusual transient giving rise to negative fluor signal
            if ysigma > 0:
                dFoY = gaussian_filter1d(dFoY, ysigma, axis=1, mode='nearest')
            tuning.append(dFoY)
            
            ## Run-to-run reliability
            for i in range(ncells):
                M = dFoY[i]  # (ybin, ntrial)
                bad_trial = np.where(M.var(axis=0)==0)[0]
                if len(bad_trial) > 0:
                    M = np.delete(M, bad_trial, axis=1)
                ntrial = M.shape[1]
                if ntrial > 1:
                    R = np.corrcoef(M.T)[np.triu_indices(ntrial, k=1)]  # All trial-pairs (upper triangular part)
                    reliability[i,count] = np.mean(R)
                else:
                    reliability[i,count] = np.NaN    
            
            count += 1
    
    # %% Remapping index (hard coded with nctx = 2)
    remapping = np.zeros((ncells, nday))
    discrimination = np.zeros((ncells, nday))
        
    for d in range(nday):
        
        Fyctx = [np.mean(tuning[2*d+c], axis=2) for c in range(nctx)]  # Trial-averaged tuning in Fam/Nov 
        Fyff = [np.mean(tuning[2*d][:,:,0::2], axis=2),
                np.mean(tuning[2*d][:,:,1::2], axis=2)]  # Split Fam tuning into even/odd trials
        Fynn = [np.mean(tuning[2*d+1][:,:,0::2], axis=2),
                np.mean(tuning[2*d+1][:,:,1::2], axis=2)]  # Split Nov tuning into even/odd trials        
        for i in range(ncells):
            ff = stats.pearsonr(Fyff[0][i], Fyff[1][i])[0]  # Fam correlation
            nn = stats.pearsonr(Fynn[0][i], Fynn[1][i])[0]  # Nov correlation
            fn = stats.pearsonr(Fyctx[0][i], Fyctx[1][i])[0]  # Fam/Nov correlation
            remapping[i,d] = np.nanmean(np.array([ff,nn])) - fn
            discrimination[i,d] = 1 - fn

    # %% Save results
    with h5py.File(datapath, 'r+') as f:
        if 'properties' in f:
            del f['properties']
        g = f.create_group('properties')
        g.create_dataset('transient_rate', data=trate)
        g.create_dataset('transient_auc', data=tauc)
        g.create_dataset('spike_rate', data=srate)
        g.create_dataset('spike_auc', data=sauc)
        g.create_dataset('spatial_info', data=si)
        g.create_dataset('reliability', data=reliability)
        g.create_dataset('remapping', data=remapping)
        g.create_dataset('discrimination', data=discrimination)
        
    # %% Boostrapping
    si_shuffle = np.zeros((n_shuffle, ncells, nday*nctx))
    count = 0
    
    for d in range(nday):
        for c in range(nctx):
            
            ks = np.where((days==(d+1)) & (ctx==c))[0]  # Day starts from 1
            position = np.hstack([data['y'][k] for k in ks])
            yin = (position>=yrange[0]) & (position<yrange[1])  # Avoid activity outside yrange increases SI after shuffling position vector...
            moving = np.hstack([data['moving'][k] for k in ks]) & yin
            trace = np.hstack([data['F'][k] for k in ks])
            transient = np.hstack([data['Tr'][k] for k in ks])
            spiking = np.hstack([data['Sp'][k] for k in ks])
            if spike:
                F = spiking[:,moving]
            else:
                F = trace[:,moving]*transient[:,moving]
                
            nframes = np.hstack([data['t'][k] for k in ks])
            nframes_ = np.hstack([0, np.cumsum(nframes)])
            ntrial = len(nframes)            
            ys = []
            for r in range(ntrial):
                seg = slice(nframes_[r], nframes_[r+1])
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
            
            print(f'Computing for day {d+1}, context {c} ...')
            with mp.Pool(processes=n_processes) as pool:    
                res = tqdm(pool.imap(si1d_wrapper, params), total=n_shuffle)
                si_shuffle[:,:,count] = np.vstack(list(res))
                
            count += 1
    
    # %% Save results
    with h5py.File(datapath, 'r+') as f:
        g = f['properties']
        if 'spatial_info_shuffle' in g:
            del f['spatial_info_shuffle']
        g.create_dataset('spatial_info_shuffle', data=si_shuffle)
        
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
    # datapaths = glob(os.path.join(r'D:\LW_Alldata\SI_Transient','*.hdf5'))
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
    
    
    
    