# -*- coding: utf-8 -*-
"""
Load invivo data (*.hdf5)

@author: Hung-Ling
"""
import numpy as np
import h5py


# %%
def get_data(datapath, mice=None, day=None, spike=False, min_rate=1/60, verbose=False):
    '''
    Parameters
    ----------
    datapath : str
    mice : int, list of int or None
        Indices of the mice to load (start from 0). If None, load all mice.
    day : int, list of int or None
        Which day to load (start from 1). If None, load all days.
    spike : bool
        Whether to use deconvolved spike data
    min_rate : float
        Threshold of active cell (select cells with transient rate >= min_rate
                                  in at least one Fam/Nov context)
    
    Returns
    -------
    data : dict, keys are 't','y','moving','F','Tr',('Sp')
        value type is list (mouse) of list (category) of arrays
    cells : dict, keys are 'trate','auc',('srate','sauc'),'si_unbiased','si_pvalue'
        value type is list (mouse) of arrays
    days : 1d array, shape (ncat,)
        Recording days indexed from 1
    ctx : 1d array, shape (ncat,)
        Corresponding context label (0: familiar, 1 novel)
    '''
    data = dict()  # Basic data
    cells = dict()  # Cell's characteristics
    
    with h5py.File(datapath, 'r') as f:
        n_mice = f.attrs['n_mice']
        ncat = f.attrs['n_categories']
        days = f.attrs['days']
        contexts = f.attrs['contexts']
        ctxnames = np.array(list(dict.fromkeys(contexts).keys()))  # ['Familiar','Novel'] Note that dict preserves order
        ctx = np.array([np.where(ctxnames==c)[0].item() for c in contexts])  # [0,1,0,1]
        data['fps'] = f.attrs['fps']
        
        if mice is None:
            mice = range(n_mice)
        elif isinstance(mice, int):
            mice = [mice]
        if day is None:
            category = np.arange(ncat)
        else:
            if isinstance(day, int):
                sub = np.where(days==day)[0]
            else:
                sub = np.where(np.logical_or.reduce([(days==d) for d in day]))[0]
            days = days[sub]
            ctx = ctx[sub]
            category = np.arange(ncat)[sub]
            ncat = len(days)
                
        for key in ['t','y','moving','F','Tr','Fn','Sp']:
            data[key] = [[] for _ in range(len(mice))]
        for key in ['trate','auc','srate','sauc','si_unbiased','si_pvalue']:
            cells[key] = [[] for _ in range(len(mice))]
            
        for n, m in enumerate(mice):
            for k in category:
                g = f[str(m)+'/'+str(k)]
                data['t'][n].append(g['n_frames'][()].astype(int))
                data['y'][n].append(g['y'][()])
                data['moving'][n].append(g['moving'][()].astype(bool))
                data['F'][n].append(g['F'][()])
                data['Tr'][n].append(g['transient'][()].astype(bool))
                if spike:
                    data['Fn'][n].append(g['Fn'][()])
                    data['Sp'][n].append(g['spike'][()])
                
            ## Cell's characteristics
            if day is None:
                sub = np.arange(2*days[-1])  # Fam/Nov 2 contexts per day
            else:
                if isinstance(day, int):
                    sub = np.arange(2*(day-1),2*day)
                else:
                    sub = np.hstack([np.arange(2*(d-1),2*d) for d in day])
            cells['trate'][n] = f[str(m)+'/transient_rate'][:,sub].T  # (nctx, ncell)
            cells['auc'][n] = f[str(m)+'/auc_rate'][:,sub].T  # (nctx, ncell)
            if spike:
                cells['srate'][n] = f[str(m)+'/spike_rate'][:,sub].T  # (nctx, ncell)
                cells['sauc'][n] = f[str(m)+'/spike_auc'][:,sub].T  # (nctx, ncell)
                
            si = f[str(m)+'/spatial_info'][:,sub]  # (ncell, nctx)
            si_shuffle = f[str(m)+'/spatial_info_shuffle'][:,:,sub]  # (n_shuffle, ncell, nctx)
            n_shuffle = si_shuffle[0].shape[0]
            si_unbiased = (si - si_shuffle.mean(axis=0)) / si_shuffle.std(axis=0)
            si_unbiased[np.isnan(si_unbiased)] = 0
            cells['si_unbiased'][n] = si_unbiased.T
            cells['si_pvalue'][n] = np.sum(si_shuffle>=si[np.newaxis,:,:], axis=0).T/n_shuffle
        
    ## Select active cells
    ncells = [data['F'][m][0].shape[0] for m in range(len(mice))]
    if spike:
        active = [srate>=min_rate for srate in cells['srate']] 
    else:
        active = [trate>=min_rate for trate in cells['trate']]
    if verbose:
        print('-'*36)
        print('Total cells: ', ncells)
        print('Active cells: ', [np.sum(np.any(a, axis=0)) for a in active])
        print('Proportion of active cells %.2f %%' % 
              (sum([np.sum(np.any(a, axis=0)) for a in active])/sum(ncells)*100))
        print('Active in both %.2f %%' % 
              (sum([np.sum(np.all(a, axis=0)) for a in active])/sum(ncells)*100))
        print('Active exclusively in Fam %.2f %%' % 
              (sum([np.sum(a[0] & ~a[1]) for a in active])/sum(ncells)*100))
        print('Active exclusively in Nov %.2f %%' % 
              (sum([np.sum(a[1] & ~a[0]) for a in active])/sum(ncells)*100))
        print('-'*36)
    
    selected = [np.any(a, axis=0) for a in active]
    if verbose:
        print('Selected cells:', selected)
    for m in range(len(mice)):
        for k in range(ncat):
            data['F'][m][k] = data['F'][m][k][selected[m],:]
            data['Tr'][m][k] = data['Tr'][m][k][selected[m],:]
            if spike:
                data['Fn'][m][k] = data['Fn'][m][k][selected[m],:]
                data['Sp'][m][k] = data['Sp'][m][k][selected[m],:]
        cells['trate'][m] = cells['trate'][m][:,selected[m]]
        cells['auc'][m] = cells['auc'][m][:,selected[m]]
        if spike:
            cells['srate'][m] = cells['srate'][m][:,selected[m]]
            cells['sauc'][m] = cells['sauc'][m][:,selected[m]]
        cells['si_unbiased'][m] = cells['si_unbiased'][m][:,selected[m]]
        cells['si_pvalue'][m] = cells['si_pvalue'][m][:,selected[m]]
    
    return data, cells, days, ctx

# %%
def get_data_bis(datapath, day=None, min_rate=1/60, select='or', verbose=False):
    '''Get invivo data of a single mouse from the hdf5 file. Cells property is per day.
    
    Parameters
    ----------
    datapath : str
        Full path to the single mouse hdf5 file.
    day : int, list of int or None
        Which day to load (start from 1). If None, load all days.
    min_rate : float
        Threshold of active cell ()
    select : 'or' (default) or 'and'
        'or' selects cells with transient rate >= min_rate in at least one Fam/Nov context
        'and' selects cells with transient rate >= min_rate in all contexts
    
    Returns
    -------
    data : dict, keys are 't','y','moving','F','Tr','Sp'
        value type is list (category) of arrays
    cells : dict, keys are 'trate','auc','srate','sauc','si_unbiased','si_pvalue'
        value type is array, shape (ncell, nctx*nday)
    days : 1d array, shape (ncat,)
        Recording days indexed from 1
    ctx : 1d array, shape (ncat,)
        Corresponding context label (0: familiar, 1 novel)
    selected_cells : 1d array, shape (ncell,)
        Indices (in the original total cell list) of selected cells
    '''
    data = dict()  # Basic data
    cells = dict()  # Cells properties
    
    with h5py.File(datapath, 'r') as f:
        ncat = f.attrs['n_categories']
        days = f.attrs['days']
        contexts = f.attrs['contexts']
        ctxnames = np.array(list(dict.fromkeys(contexts).keys()))  # ['Familiar','Novel'] Note that dict preserves order
        ctx = np.array([np.where(ctxnames==c)[0].item() for c in contexts])  # [0,1,0,1]
        data['fps'] = f.attrs['fps']
        nday = len(set(days))
        nctx = len(set(ctx))
        
        if day is None:
            sub = np.ones(ncat, dtype=bool)
        else:
            if isinstance(day, int):
                sub = (days==day)
            else:
                sub = np.any(np.vstack([days==d for d in day]), axis=0)
            days = days[sub]
            ctx = ctx[sub]
        category = np.arange(ncat)[sub]
        ncat = len(days)
                
        for key in ['t','y','moving','F','Tr','Sp']:
            data[key] = [[] for _ in range(ncat)]
        
        for j, k in enumerate(category):
            g = f[str(k)]
            data['t'][j] = g['n_frames'][()].astype(int)
            data['y'][j] = g['y'][()]
            data['moving'][j] = g['moving'][()].astype(bool)
            data['F'][j] = g['F'][()]
            data['Tr'][j] = g['transient'][()].astype(bool)
            data['Sp'][j] = g['spike'][()]
            
        ## Cells properties
        ctx_day = np.repeat(np.arange(1,nday+1), nctx)
        if day is None:
            ind = np.ones(nday*nctx, dtype=bool)
        else:
            if isinstance(day, int):
                ind = (ctx_day==day)
            else:
                ind = np.any(np.vstack([ctx_day==d for d in day]), axis=0)
        g = f['properties']
        cells['trate'] = g['transient_rate'][:,ind]  # (ncell, nctx*nday)
        cells['tauc'] = g['transient_auc'][:,ind]
        cells['srate'] = g['spike_rate'][:,ind]
        cells['sauc'] = g['spike_auc'][:,ind]
        cells['reliability'] = g['reliability'][:,ind]
         
        cells['remapping'] = g['remapping'][()] if day is None else g['remapping'][:,np.array(day)-1]  # Note that day starts from 1 
        cells['discrimination'] = g['discrimination'][()] if day is None else g['discrimination'][:,np.array(day)-1]
        
        si = g['spatial_info'][:,ind]  # (ncell, nctx*nday)
        si_shuffle = g['spatial_info_shuffle'][:,:,ind]  # (n_shuffle, ncell, nctx*nday)
        n_shuffle = si_shuffle.shape[0]
        shuffle_std = np.std(si_shuffle, axis=0)  # (ncell, nctx*nday)
        shuffle_nan = (shuffle_std == 0)
        si_unbiased = (si - si_shuffle.mean(axis=0))
        si_unbiased[shuffle_nan] = 0  # np.NaN
        si_unbiased[~shuffle_nan] /= shuffle_std[~shuffle_nan]
        cells['si_unbiased'] = si_unbiased  # (ncell, nctx*nday)
        cells['si_pvalue'] = np.sum(si_shuffle>=si[np.newaxis,:,:], axis=0)/n_shuffle
    
    ## Select active cells
    ncell = data['F'][0].shape[0]
    
    rate = cells['trate']  # (ncell, nctx*nday)
    # rate = cells['srate'] if spike else cells['trate']
    active = (rate > min_rate).T  # (nctx*nday, ncell)
    # day_ctx = np.tile(np.arange(nctx), len(set(days)))  # [0,1,0,1,0,1,...]
    # active = np.zeros((nctx, ncell), dtype=bool)
    # for c in range(nctx):
    #     active[c] = (np.mean(rate[:,day_ctx==c], axis=1) >= min_rate)
    
    if select == 'or':    
        selected = np.any(active, axis=0)
    elif select == 'and':
        selected = np.all(active, axis=0)
    else:
        selected = np.ones(active.shape[1], dtype=bool)
    selected_cells = np.where(selected)[0]
    ncell_active = len(selected_cells)
    AC = np.array([np.sum(np.all(~active, axis=0)),  # None
                   np.sum(active[0] & ~active[1]),  # Exclusively Fam
                   np.sum(np.all(active, axis=0)),  # Both
                   np.sum(active[1] & ~active[0])])  # Exclusively Nov
    if verbose:
        print('-'*36)
        print('Total cells:', ncell)
        print('Active cells:', ncell_active)
        print('Proportion of active cells %.2f %%' % (ncell_active/ncell*100))
        print('Active in both %.2f %%' % (AC[2]/ncell*100))
        print('Active exclusively in Fam %.2f %%' % (AC[1]/ncell*100))
        print('Active exclusively in Nov %.2f %%' % (AC[3]/ncell*100))
        print('-'*36)
        
    ## Take only active cells
    for k in range(ncat):
        for key in ['F','Tr','Sp']:
            data[key][k] = data[key][k][selected]
    for key in cells.keys():
        cells[key] = cells[key][selected]
    
    return data, cells, days, ctx, selected_cells

# %%
def get_cells_session(datapath, day=None, min_rate=1/60):
    '''Get cells property per session from the invivo hdf5 file.
    '''
    cells = dict()
    with h5py.File(datapath, 'r') as f:
        ncat = f.attrs['n_categories']
        days = f.attrs['days']
        contexts = f.attrs['contexts']
        ctxnames = np.array(list(dict.fromkeys(contexts).keys()))  # ['Familiar','Novel'] Note that dict preserves order
        ctx = np.array([np.where(ctxnames==c)[0].item() for c in contexts])  # [0,1,0,1]
        rate = f['properties/transient_rate'][()]  # (ncell, nday*nctx)
        
    ## Select active cells
    if min_rate > 0:
        nday = len(set(days))
        nctx = len(set(ctx))
        ctx_day = np.repeat(np.arange(1,nday+1), nctx)  # [1,1,2,2,3,3,...]
        if day is not None:
            if isinstance(day, int):
                ind = (ctx_day==day)
            else:
                ind = np.any(np.vstack([ctx_day==d for d in day]), axis=0)
            rate = rate[:,ind]    
        active = (rate >= min_rate).T  # (nday*nctx, ncell)
        # active = np.vstack([rate[:,c::2].mean(axis=1) for c in range(nctx)])  # (nctx, ncell)
        selected = np.any(active, axis=0)
    else:
        selected = np.ones(rate.shape[0], dtype=bool)
        
    if day is None:
        sub = np.ones(ncat, dtype=bool)
    else:
        if isinstance(day, int):
            sub = (days==day)
        else:
            sub = np.any(np.vstack([days==d for d in day]), axis=0)
    
    with h5py.File(datapath, 'r') as f:
        cells['trate'] = f['transient_rate'][selected,:][:,sub]  # (ncell, ncat)
        cells['tauc'] = f['transient_auc'][selected,:][:,sub]
        cells['srate'] = f['spike_rate'][selected,:][:,sub]
        cells['sauc'] = f['spike_auc'][selected,:][:,sub]
        cells['reliability'] = f['reliability'][selected,:][:,sub]
        si = f['spatial_info'][selected,:][:,sub]
        si_shuffle = f['spatial_info_shuffle'][:,selected,:][:,:,sub]  # (n_shuffle, ncell, ncat)
        n_shuffle = si_shuffle.shape[0]
        shuffle_std = np.std(si_shuffle, axis=0)  # (ncell, ncat)
        shuffle_nan = (shuffle_std == 0)
        si_unbiased = (si - si_shuffle.mean(axis=0))
        si_unbiased[shuffle_nan] = 0  # np.NaN
        si_unbiased[~shuffle_nan] /= shuffle_std[~shuffle_nan]
        cells['si_unbiased'] = si_unbiased  # (ncell, nctx*nday)
        cells['si_pvalue'] = np.sum(si_shuffle>=si[np.newaxis,:,:], axis=0)/n_shuffle
        
    return cells
