# -*- coding: utf-8 -*-
"""
Functions to evaluate the prediction accuracy.

"""
import numpy as np

# %%
def get_majority(y_test, y_pred):
    '''
    Majority score of a binary classification problem.

    Parameters
    ----------
    y_test : np.ndarray of shape (nsample,)
    y_pred : np.ndarray of shape (nsample,)
    
    Returns
    -------
    float : 1 (correct)|0.5 (undetermined)|0 (incorrect)
    '''
    accuracy = np.mean(y_test==y_pred)
    if accuracy > 0.5:
        return 1.0
    elif accuracy == 0.5:
        return 0.5
    else:
        return 0.0
    
# %% R-squared (R2)
def get_R2(y_test, y_pred):
    '''
    Function calculating the coefficient of determination,
    i.e. R-squared, fraction of variance explained.

    Parameters
    ----------
    y_test : np.ndarray of shape (nsample, nfeature)
        The true outputs.
    y_pred : np.ndarray of shape (nsample, nfeature)
        The predicted outputs

    Returns
    -------
    R2 : np.ndarray of shape (nfreature,)
        R-squared for each output feature
    '''
    if y_test.ndim == 1:
        y_test = y_test[:, np.newaxis]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]
    R2_list = []  # Initialize a list that will contain the R2s for all the outputs
    for i in range(y_test.shape[1]):
        y_mean = np.mean(y_test[:,i])
        R2 = 1 - np.sum((y_pred[:,i]-y_test[:,i])**2)/np.sum((y_test[:,i]-y_mean)**2)
        R2_list.append(R2)
    return np.array(R2_list)

# %% Pearson's correlation coefficient (rho)
def get_rho(y_test, y_pred):
    '''
    Function calculating Pearson's correlation coefficient (rho).

    Parameters
    ----------
    y_test : np.ndarray of shape (nsample, nfeature)
        The true outputs.
    y_pred : np.ndarray of shape (nsample, nfeature)
        The predicted outputs

    Returns
    -------
    rho : np.ndarray of shape (nfreature,)
        An array of rho's for each output feature
    '''
    if y_test.ndim == 1:
        y_test = y_test[:, np.newaxis]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]
    rho_list = []  # Initialize a list that will contain the rhos for all the outputs
    for i in range(y_test.shape[1]): 
        rho = np.corrcoef(y_test[:,i].T, y_pred[:,i].T)[0,1]
        rho_list.append(rho)
    return np.array(rho_list)

# %%
def get_error(y_test, y_pred, kind='median'):
    
    if y_test.ndim == 1:
        y_test = y_test[:, np.newaxis]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]
    distances = np.sqrt(np.sum((y_test - y_pred)**2, axis=1))
    if kind == 'mean':
        return np.mean(distances)
    elif kind == 'median':
        return np.median(distances)
    elif kind == 'max':
        return np.max(distances)
    
        
        
        
        
        
        