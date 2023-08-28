# -*- coding: utf-8 -*-
"""
Useful functions

@author: Hung-Ling
"""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
from scipy.ndimage import gaussian_filter1d
from scikit_posthocs import posthoc_dunn

# %%
def sort_tuning(F, ysigma=0):
    '''
    Parameters
    ----------
    F : np.ndarray (2D), shape (ncell, ybin)
        Tuning curves of a neuron population.
    ysigma : float
        1D Gaussina kernel used to smooth the tuning curves.
        The default is 0 (no smoothing).

    Returns
    -------
    np.ndarray (2D), shape (ncell, ybin)
        Tuning curves with sorted neuron population.
    sort_idx : np.ndarray (1D), shape (ncell,)
        Sorted neuron indices.
    '''
    F2 = F.copy()
    F2[np.isnan(F)] = 0
    if ysigma > 0:
        F2 = gaussian_filter1d(F2, sigma=ysigma, mode='nearest', axis=1)
        
    max_pos = np.argmax(F2, axis=1)
    sort_idx = np.argsort(max_pos)
    
    return F[sort_idx,:], sort_idx

# %%
def discard_nan(nested_list):
    
    Na = len(nested_list)
    data_list = [[] for _ in range(Na)]
    for m, data in enumerate(nested_list):
        if isinstance(data, list):
            data = np.row_stack(data)  # Shape (Nb, Nc)
        ind = np.all(np.isfinite(data), axis=0)
        data_list[m] = data[:,ind]
    return data_list

def discard_zero(nested_list):
    
    Na = len(nested_list)
    data_list = [[] for _ in range(Na)]
    for m, data in enumerate(nested_list):
        if isinstance(data, list):
            data = np.row_stack(data)  # Shape (Nb, Nc)
        ind = np.any(data==0, axis=0)
        data_list[m] = data[:,~ind]
    return data_list

def long_dataframe(nested_list, varnames=['A','B','C'], varvalues=[None,None,None]):
    '''
    Construct a long-form pandas dataframe from a nested list of data consisting of
    3 variables ('A','B','C').
    This format is suited for seaborn grouped categorical plot with x='A' (categorical),
    hue='B' (categorical) and y='C' (continuous values).
    
    Parameters
    ----------
    nested_list : list of list of 1d array or list of 2d array (B-by-C)
        3-dimensional data structure consisting of two categorical variables
        (list A of list B) and a continuous variable (array C)        
    varnames : list of 3 entries
        Names of the variables. The default is ['A','B','C']
    varvalues : list of 3 entries
        Possible values of each variable. The default is [None,None,None].
        If None for 'A' or 'B', the values are taken in e.g., [0,1,...,|A|-1]
        None for 'C' as a continuous variable.

    Returns
    -------
    df : pandas.DataFrame
        Long format of the 3-dimensional data structure.
    '''
    df_list = []
    for a, la in enumerate(nested_list):
        for b, lb in enumerate(la):
            xc = np.array(lb)
            if varvalues[0] is None:
                xa = np.repeat(a, len(xc))
            else:
                xa = np.repeat(varvalues[0][a], len(xc))
            if varvalues[1] is None:
                xb = np.repeat(b, len(xc))
            else:
                xb = np.repeat(varvalues[1][b], len(xc))
            df_list.append(pd.DataFrame({varnames[0]: xa,
                                         varnames[1]: xb,
                                         varnames[2]: xc}))
    df = pd.concat(df_list, ignore_index=True)
    return df

# %%
def compare_grouped(data_list, varnames=['A','B','C'], varvalues=[None,None,None],
                    fig=None, ax=None, kind='box', test='Mann-Whitney', **kwargs):
    '''
    Grouped boxplot (kind='box'), barplot (kind='bar'), or violinplot (kind='violin')
    and annotate the statistical significance using test in 
    ['t-test_ind','t-test_welch','t-test_paired','Mann-Whitney','Mann-Whitney-gt',
     'Mann-Whitney-ls','Levene','Wilcoxon','Kruskal'].
    
    See also long_dataframe function for input data structure.
    '''
    fsize = plt.rcParams['font.size']
    lwidth = plt.rcParams['axes.linewidth']
    Na = len(data_list)
    Nb = len(data_list[0])  # Normally Nb = 2
    if varvalues[0] is None:
        varvalues[0] = np.arange(Na)
    if varvalues[1] is None:
        varvalues[1] = np.arange(Nb)
    if fig is None:
        fig = plt.figure(figsize=(1.6+Na*0.8,5))
    if ax is None:
        ax = plt.gca()
        
    if test in ['t-test_paired','Wilcoxon']:  # Dependent samples, NaN is not allowed
        data_list = discard_nan(data_list)
    df = long_dataframe(data_list, varnames=varnames, varvalues=varvalues)
    
    if kind == 'box':
        sns.boxplot(data=df, x=varnames[0], y=varnames[2], hue=varnames[1], ax=ax, **kwargs)
    elif kind == 'bar':
        sns.barplot(data=df, x=varnames[0], y=varnames[2], hue=varnames[1], ax=ax, **kwargs)
    elif kind == 'violin':
        sns.violinplot(data=df, x=varnames[0], y=varnames[2], hue=varnames[1], ax=ax, **kwargs) 
        
    pformat = {'pvalue_thresholds':[[1e-3,'***'],[0.01,'**'],[0.05,'*'],[1,'ns']], 'fontsize':fsize}
    pairs = [((v,varvalues[1][0]),(v,varvalues[1][1])) for v in varvalues[0]]
    annot = Annotator(ax, pairs, data=df, x=varnames[0], y=varnames[2], hue=varnames[1])  # verbose=False
    annot.configure(test=test, loc='outside', line_height=0., line_width=lwidth, pvalue_format=pformat)
    annot.apply_and_annotate()
    
    ax.set_xticklabels(varvalues[0])
    ax.set_ylabel(varnames[2])
    ax.legend(loc='best')
    fig.tight_layout()
    
    return fig, ax

# %%
def compare_paired(data_list, varnames=['A','B','C'], varvalues=[None,None,None],
                   fig=None, ax=None, palette=['C0','C1'], test='Wilcoxon', **kwargs):
    '''
    Grouped box plot (matplotlib) with overlayed stripplot (seaborn) and annotate 
    the statistical significance with paired t-test or Wilcoxon signed-rank test
    (dependent samples).
    
    See also long_dataframe function for input data structure.
    '''
    fsize = plt.rcParams['font.size']
    lwidth = plt.rcParams['axes.linewidth']
    Na = len(data_list)
    Nb = len(data_list[0])  # Normally Nb = 2
    if varvalues[0] is None:
        varvalues[0] = np.arange(Na)
    if varvalues[1] is None:
        varvalues[1] = np.arange(Nb)
    if fig is None:
        fig = plt.figure(figsize=(1.6+Na*0.8,5))
    if ax is None:
        ax = plt.gca()
        
    data_list = discard_nan(data_list)
    df = long_dataframe(data_list, varnames=varnames, varvalues=varvalues)
     
    ## Draw box plot
    ff = 0.8  # Fill factor (total bar width, max = 1)
    width = ff/Nb  # Bar width
    for b in range(Nb):
        pos = np.arange(Na) + (-ff/2 + width/2 + b*width)
        ax.boxplot([R[b] for R in data_list], positions=pos, widths=width,
                   showfliers=False, showbox=False, showcaps=False,
                   whiskerprops=dict(lw=0), medianprops=dict(lw=3, c=palette[b]))

    sns.stripplot(data=df, x=varnames[0], y=varnames[2], hue=varnames[1], ax=ax,
                  palette=palette, **kwargs)  # dodge=True, alpha=0.8
    
    pformat = {'pvalue_thresholds':[[1e-3,'***'],[0.01,'**'],[0.05,'*'],[1,'ns']], 'fontsize':fsize}
    pairs = [((v,varvalues[1][0]),(v,varvalues[1][1])) for v in varvalues[0]]
    annot = Annotator(ax, pairs, data=df, x=varnames[0], y=varnames[2], hue=varnames[1])  # verbose=False
    annot.configure(test=test, loc='outside', line_height=0., line_width=lwidth, pvalue_format=pformat) 
    annot.apply_and_annotate()
    
    ax.set(xlim=[-0.5, Na-0.5], xticks=np.arange(Na), xticklabels=varvalues[0])
    ax.legend(loc='best')
    fig.tight_layout()
    
    return fig, ax
    
# %%
def grouped_pair_lines(data_list, varnames=['A','B','C'], varvalues=[None,None,None],
                       fig=None, ax=None, palette=['C0','C1'], pairs=None, test='t-test_paired'):
    '''
    Grouped bar plot (mean and s.e.m.) with paired lines and annotate the statistical
    significance with paired t-test or Wilcoxon signed-rank test (if pairs are provided).
    
    See also long_dataframe function for input data structure.
    '''
    fsize = plt.rcParams['font.size']
    lwidth = plt.rcParams['axes.linewidth']
    Na = len(data_list)
    Nb = len(data_list[0])  # Normally Nb = 2
    if varvalues[0] is None:
        varvalues[0] = np.arange(Na)
    if varvalues[1] is None:
        varvalues[1] = np.arange(Nb)
    if fig is None:
        fig = plt.figure(figsize=(1.6+Na*0.8,5))
    if ax is None:
        ax = plt.gca()
        
    data_list = discard_nan(data_list)
    df = long_dataframe(data_list, varnames=varnames, varvalues=varvalues)
    
    ## Draw paired lines
    ff = 0.8  # Fill factor (total bar width, max = 1)
    width = ff/Nb  # Bar width
    for a in range(Na):
        pos = np.arange(a-ff/2+width/2, a+ff/2, width)
        ax.plot(pos, data_list[a], lw=1.2, c='gray', alpha=0.8, zorder=-1)  # # marker='o', ms=4, mfc='none'
        
    ## Draw bar plot
    for b in range(Nb):
        x = np.arange(Na) + (-ff/2 + width/2 + b*width)
        height = [np.mean(R[b]) for R in data_list]
        yerr = [stats.sem(R[b]) for R in data_list]
        ax.bar(x, height, yerr=yerr, width=width, color='none', linewidth=lwidth, 
               edgecolor=palette[b], ecolor=palette[b], capsize=4,
               error_kw=dict(elinewidth=lwidth, capthick=lwidth))
    
    if pairs is not None:
        pformat = {'pvalue_thresholds':[[1e-3,'***'],[1e-2,'**'],[0.05,'*'],[1,'ns']], 'fontsize':fsize}
        annot = Annotator(ax, pairs, data=df, x=varnames[0], y=varnames[2], hue=varnames[1])
        annot.configure(test=test, loc='outside', line_height=0., line_width=lwidth, pvalue_format=pformat)
        annot.apply_and_annotate()
    
    ax.set_xticks(np.arange(Na))
    ax.set_xticklabels(varvalues[0])
    fig.tight_layout()
    
    return fig, ax

# %%
def compare_multiple(data_list, category=['A','B','C'], palette=['C0','C1','C2'],
                     fig=None, ax=None, kind='box', show_data=True, annotate=True,
                     test='anova', post_hoc='tukey'):
    
    fsize = plt.rcParams['font.size']
    lwidth = plt.rcParams['axes.linewidth']
    N = len(data_list)  # Typically N = 3
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = plt.gca()
    
    ## Plotting
    df = pd.DataFrame({'Data': np.hstack(data_list),
                       'Category': np.hstack([np.repeat(category[p], len(data_list[p]))
                                              for p in range(N)])})
    if kind == 'bar':
        for p in range(N):
            height = np.mean(data_list[p])
            yerr = stats.sem(data_list[p])
            ax.bar(p, height, yerr=yerr, width=0.65, color='none', linewidth=lwidth, 
                   edgecolor=palette[p], ecolor=palette[p], capsize=4,
                   error_kw=dict(elinewidth=lwidth, capthick=lwidth))
    elif kind == 'box':
        sns.boxplot(data=df, x='Category', y='Data', ax=ax, palette=palette, 
                    width=0.6, whis=(1,99), showfliers=False)
    elif kind == 'violin':
        sns.violinplot(data=df, x='Category', y='Data', ax=ax, palette=palette,
                       cut=0, inner='box')
    if show_data:
        sns.stripplot(data=df, x='Category', y='Data', hue='Category', ax=ax,
                      palette=palette, size=6, jitter=0.1)
    
    ## Statistical test
    if test == 'anova' and post_hoc == 'tukey':
        pval = stats.f_oneway(*data_list)[1]
        print('One-way ANOVA test, p-value: %.6f' % pval)
        pval_paired = stats.tukey_hsd(*data_list).pvalue
        print('Post-hoc Tukey HSD test\'s test')
    elif test == 'kruskal' and post_hoc == 'dunn':
        pval = stats.kruskal(*data_list, nan_policy='omit')[1]
        print('Kruskal-Wallis test, p-value: %.6f' % pval)
        pval_paired = posthoc_dunn(data_list, p_adjust='bonferroni').to_numpy()  # fdr_bh
        print('Post-hoc Dunn\'s test')
    else:
        return fig, ax
    
    pairs, pvalues = [], []
    for u, v in zip(*np.triu_indices(N, k=1)):
        pairs.append((category[u], category[v]))
        pvalues.append(pval_paired[u,v])
        print(f'{category[u]} vs {category[v]} p-value: {pval_paired[u,v]:.6f}')
        
    ## Statistical annotation
    if annotate:
        pformat = {'pvalue_thresholds':[[1e-3,'***'],[1e-2,'**'],[0.05,'*'],[1,'ns']], 'fontsize':fsize}
        annot = Annotator(ax, pairs, data=df, x='Category', y='Data', verbose=False)
        annot.configure(test=None, loc='outside', line_width=lwidth, line_height=0., pvalue_format=pformat)
        annot.set_pvalues(pvalues)
        annot.annotate()
    
    return fig, ax
