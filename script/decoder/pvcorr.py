# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 17:53:00 2022

@author: Hung-Ling
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

from function.tuning_function import tuning_1d

# %%
class PVCorr(object):
    def __init__(self, yrange=(0.1,2.1), ybin=80, ysigma=1):
        self.yrange = yrange
        self.ybin = ybin
        self.ysigma = ysigma
        
    def fit(self, X_train, y_train):
        ## Compute tuing functions
        occupancy, tuning = tuning_1d(X_train.T, y_train, yrange=self.yrange, ybin=self.ybin)
        occupancy += 1  # Avoid zero occupancy
        occupancy = gaussian_filter1d(occupancy.astype(float), self.ysigma, mode='nearest')
        self.occupancy = occupancy/np.sum(occupancy)  # P(Y)
        
        tuning[np.isnan(tuning)] = 0
        tuning[tuning < 0] = 0
        tuning = gaussian_filter1d(tuning, self.ysigma, axis=1, mode='nearest')
        self.tuning = tuning
                
    def predict(self, X_test):
        y_min, y_max = self.yrange
        y_edges = np.linspace(y_min, y_max, self.ybin+1)
        y_centers = (y_edges[:-1] + y_edges[1:])/2
        
        nt, ncell = X_test.shape
        y_pred = np.empty(nt)
        
        for t in range(nt):
            rs = X_test[t,:]
            if np.sum(rs) == 0:
                y_pred[t] = y_centers[np.argmax(self.occupancy)]
                # print('Encounter zero vector at point %d, take the max occupancy' % t)
            else:
                corr = np.corrcoef(rs, self.tuning, rowvar=False)[0,1:]
                y_pred[t] = y_centers[np.argmax(corr)]
                
        return y_pred
    
    
    