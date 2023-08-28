# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 15:57:50 2021

Adapted codes from Neural_Decoding
@author: Hung-Ling 
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

from function.tuning_function import tuning_1d
from function.shape import procrustes

# %% Decoder class
class NaiveBayes1D(object):
    def __init__(self, yrange=(0.1,2.1), ybin=80, ysigma=4, prior=False):
        self.yrange = yrange
        self.ybin = ybin
        self.ysigma = ysigma
        self.prior = prior
        
    def fit(self, X_train, y_train):
        ## Compute tuing functions
        ncell = X_train.shape[1]
        
        occupancy, tuning = tuning_1d(X_train.T, y_train, yrange=self.yrange, ybin=self.ybin)
        occupancy += 1  # Avoid zero occupancy
        occupancy = gaussian_filter1d(occupancy.astype(float), self.ysigma, mode='nearest')
        self.occupancy = occupancy/np.sum(occupancy)  # P(Y)
        
        self.tuning = np.ones((ncell, self.ybin))/self.ybin  # Initially uniform distribution
        tuning[np.isnan(tuning)] = 0
        tuning[tuning < 0] = 0
        tuning = gaussian_filter1d(tuning, self.ysigma, axis=1, mode='nearest')
        for i, curve in enumerate(tuning):
            a = np.sum(curve)
            if a > 0:
                self.tuning[i] = curve/a
        
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
                tuning_ = self.tuning*rs[:,np.newaxis]
                ## Total likelihood as the product of all probability distribution
                ## Avoid numerical underflow of np.prod(tuning_, axis=0):
                likelihood = np.exp(np.sum(np.log(tuning_+1), axis=0))-1  # Approximation
                if self.prior:
                    y_pred[t] = y_centers[np.argmax(likelihood*self.occupancy)]
                else:
                    y_pred[t] = y_centers[np.argmax(likelihood)]
            
        return y_pred

# %% Decoder class with the Procrustes transformation
class DecodeNovel(object):
    def __init__(self, yrange=(0.1,2.1), ybin=80, ysigma=1, prior=False):
        self.yrange = yrange
        self.ybin = ybin
        self.ysigma = ysigma
        self.prior = prior
        
    def fit(self, X_train, y_train):
        ## Compute tuing functions (Familiar and Novel)
        self.tuning = [[],[]]
        for c in range(2):
            ncell = X_train[c].shape[1]
            
            occupancy, tuning = tuning_1d(X_train[c].T, y_train[c], yrange=self.yrange, ybin=self.ybin)
            if c == 1:  # Novel
                occupancy += 1  # Avoid zero occupancy
                occupancy = gaussian_filter1d(occupancy.astype(float), self.ysigma, mode='nearest')
                self.occupancy = occupancy/np.sum(occupancy)  # P(Y)
            
            self.tuning[c] = np.ones((ncell, self.ybin))/self.ybin  # Initially uniform distribution
            tuning[np.isnan(tuning)] = 0
            tuning[tuning < 0] = 0
            tuning = gaussian_filter1d(tuning, self.ysigma, axis=1, mode='nearest')
            for i, curve in enumerate(tuning):
                a = np.sum(curve)
                if a > 0:
                    self.tuning[c][i] = curve/a
            
        ## Compute the Procrustes transformation
        (A, B), (R, s), _ = procrustes(self.tuning[0].T, self.tuning[1].T, centered=False)
        self.tuning_aligned = [A.T, B.T]
        self.transformation = [R, s]
        
    def predict(self, X_test):
        y_min, y_max = self.yrange
        y_edges = np.linspace(y_min, y_max, self.ybin+1)
        y_centers = (y_edges[:-1] + y_edges[1:])/2
        
        nt, ncell = X_test.shape
        y_pred = np.empty(nt)
        
        X_transformed = X_test @ self.transformation[0].T
        X_transformed[X_transformed < 0] = 0  # Nonnegative activity
        self.X_transformed = X_transformed  
        
        for t in range(nt):
            rs = X_transformed[t,:]
            if np.sum(rs) == 0:
                y_pred[t] = y_centers[np.argmax(self.occupancy)]
                # print('Encounter zero vector at point %d, take the max occupancy' % t)
            else:
                tuning_ = self.tuning[0]*rs[:,np.newaxis]  # Apply on Familiar tuning map
                 ## Total likelihood as the product of all probability distribution
                ## Avoid numerical underflow of np.prod(tuning_, axis=0):
                likelihood = np.exp(np.sum(np.log(tuning_+1), axis=0))-1  # Approximation
                if self.prior:
                    y_pred[t] = y_centers[np.argmax(likelihood*self.occupancy)]
                else:
                    y_pred[t] = y_centers[np.argmax(likelihood)]
            
        return y_pred        
         
        
        
        