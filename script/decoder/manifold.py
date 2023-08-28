# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 00:01:21 2022

@author: Hung-Ling
"""
import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
import umap

from decoder.pvcorr import PVCorr
from decoder.bayesian import NaiveBayes1D

# %%
def get_landmark(X, y, yrange=(0.,4.), ybin=80):
    
    y_in = (y >= yrange[0]) & (y < yrange[1])
    X2 = X.copy()[y_in,:]
    y2 = y.copy()[y_in]
    
    y_edges = np.linspace(*yrange, ybin+1)
    jy = np.searchsorted(y_edges, y2, side='right')-1
    
    points = [[] for _ in range(ybin)]
    for i, j in enumerate(jy):
        points[j].append(X2[i])
    landmark = [[] for _ in range(ybin)]
    for j in range(ybin):
        landmark[j] = np.vstack(points[j]).mean(axis=0)
    landmark = np.vstack(landmark)
    
    return landmark

# %%
def shape_size(X, normalize=False):
    
    if normalize:
        return np.sqrt(np.sum(X**2)/X.shape[0])
    else:
        return np.sqrt(np.sum(X**2))  # Same as Frobenius norm: np.linalg.norm(X)

def normalize_shape(X):
    
    center = np.mean(X, axis=0)
    X2 = X - center[np.newaxis,:]
    scale = np.mean(np.sqrt(np.sum(X2**2, axis=1)))  # Averaged euclidean distance
    X2 /= scale
    
    return X2, center, scale

def disparity(A, B):
    
    return np.mean(np.sqrt(np.sum((A - B)**2, axis=1)))
    
# %% Spatial decoder
class SpatialDecoder(object):
    def __init__(self, reduction='none', dim=2, method='knr', k=5, knr_metric='cosine',
                 yrange=(0.,4.), ybin=80, ysigma=1, normalize=False, **param_umap):
        self.reduction = reduction
        self.dim = dim
        self.method = method
        self.k = k  # Number of neighbors in KNeighborsRegressor
        self.knr_metric = knr_metric  # Metric used in KNeighborsRegressor
        self.yrange = yrange
        self.ybin = ybin
        self.ysigma = ysigma
        self.normalize = normalize
        self.param_umap = param_umap
        
    def fit(self, X_train, y_train):
        if self.reduction == 'none':
            self.X_train_embed = X_train.copy()
        elif self.reduction == 'pca':
            self.reducer = PCA(n_components=self.dim).fit(X_train)
            self.X_train_embed = self.reducer.transform(X_train)
        elif self.reduction == 'umap':
            self.reducer = umap.UMAP(n_components=self.dim, **self.param_umap).fit(X_train)
            self.X_train_embed = self.reducer.embedding_.copy()
        
        ## Normalize manifold (center and scale to unit norm)
        if self.normalize:
            self.X_train_norm, _, _ = normalize_shape(self.X_train_embed)
        else:
            self.X_train_norm = self.X_train_embed
        
        if self.method == 'knr':
            self.decoder = KNeighborsRegressor(
                n_neighbors=self.k, metric=self.knr_metric)  # weights='distance'
        elif self.method == 'gpr':
            self.decoder = GaussianProcessRegressor()
        elif self.method == 'svr':
            self.decoder = SVR(kernel='linear')
        elif self.method == 'pvcorr':
            self.decoder = PVCorr(yrange=self.yrange, ybin=self.ybin, ysigma=self.ysigma)
        elif self.method == 'bayesian':
            self.decoder = NaiveBayes1D(yrange=self.yrange, ybin=self.ybin, ysigma=self.ysigma)
        self.decoder.fit(self.X_train_norm, y_train)
        
    def predict(self, X_test):
        if self.reduction == 'none':
            self.X_test_embed = X_test.copy()
        else:
            self.X_test_embed = np.copy(self.reducer.transform(X_test))
        
        if self.normalize:
            self.X_test_norm, _, _ = normalize_shape(self.X_test_embed)
        else:
            self.X_test_norm = self.X_test_embed

        ## Discard NaN (due to disconnected points ?)
        ind = np.all(np.isfinite(self.X_test_norm), axis=1)
        
        y_pred = np.repeat(np.NaN, self.X_test_norm.shape[0])
        y_pred[ind] = self.decoder.predict(self.X_test_norm[ind,:])
        
        return y_pred

# %% Spatial decoder with two aligned manifolds (all trials to dimension reduction)
class SpatialNovelDecoder(object):
    def __init__(self, reduction='none', dim=2, method='knr', k=5, knr_metric='cosine',
                 yrange=(0.,4.), ybin=80, ysigma=1, **param_umap):
        self.reduction = reduction
        self.dim = dim
        self.method = method
        self.k = k  # Number of neighbors in KNeighborsRegressor
        self.knr_metric = knr_metric  # Metric used in KNeighborsRegressor
        self.yrange = yrange
        self.ybin = ybin
        self.ysigma = ysigma
        self.param_umap = param_umap    
        
    def fit(self, X_train, y_train):
        
        ## Manifold learn
        if self.reduction == 'none':
            X_train_embed0 = X_train[0].copy()
            X_train_embed1 = X_train[1].copy()
        elif self.reduction == 'pca':
            self.reducer0 = PCA(n_components=self.dim).fit(X_train[0])
            self.reducer1 = PCA(n_components=self.dim).fit(X_train[1])
            X_train_embed0 = self.reducer0.transform(X_train[0])
            X_train_embed1 = self.reducer1.transform(X_train[1])
        elif self.reduction=='umap':
            self.reducer0 = umap.UMAP(n_components=self.dim, **self.param_umap).fit(X_train[0])
            self.reducer1 = umap.UMAP(n_components=self.dim, **self.param_umap).fit(X_train[1])
            X_train_embed0 = self.reducer0.embedding_.copy()  # Important!! copy array!!
            X_train_embed1 = self.reducer1.embedding_.copy()
            
        ## Normalize manifolds
        self.centers = [X_train_embed0.mean(axis=0), X_train_embed1.mean(axis=0)]
        self.scales = [shape_size(X_train_embed0 - self.centers[0], normalize=True),
                       shape_size(X_train_embed1 - self.centers[1], normalize=True)]
        self.manifolds_train = [(X_train_embed0 - self.centers[0]) / self.scales[0],
                                (X_train_embed1 - self.centers[1]) / self.scales[1]]
        ## Get landmark points
        self.landmarks = [get_landmark(self.manifolds_train[0], y_train[0], yrange=self.yrange, ybin=self.ybin),
                          get_landmark(self.manifolds_train[1], y_train[1], yrange=self.yrange, ybin=self.ybin)]
        
        ## Align manifolds without scaling (rotation/reflection)
        R, s = orthogonal_procrustes(*self.landmarks)
        self.transformation = R
        self.landmarks_aligned = [self.landmarks[0], self.landmarks[1] @ R.T]
        self.manifolds_aligned = [self.manifolds_train[0], self.manifolds_train[1] @ R.T]
        
        ## Train position decoder
        if self.method == 'knr':
            self.decoder = KNeighborsRegressor(
                n_neighbors=self.k, metric=self.knr_metric)  # weight='distance'
        elif self.method == 'gpr':
            self.decoder = GaussianProcessRegressor()
        elif self.method == 'svr':
            self.decoder = SVR(kernel='linear')
        elif self.method == 'pvcorr':
            self.decoder = PVCorr(yrange=self.yrange, ybin=self.ybin, ysigma=self.ysigma)
        elif self.method == 'bayesian':
            self.decoder = NaiveBayes1D(yrange=self.yrange, ybin=self.ybin, ysigma=self.ysigma)
        self.decoder.fit(self.manifolds_train[0], y_train[0])
            
    def predict(self, X_test):
        if self.reduction == 'none':
            X_test_embed = X_test
        else:
            X_test_embed = self.reducer1.transform(X_test)
        self.manifold_test = X_test_embed.copy()  # Important!! copy array!!
        
        ## Possibly contain NaN (?), replace with the centroid
        # good_samples = np.all(np.isfinite(self.manifold_test), axis=1)
        # bad_samples = ~good_samples
        # if np.any(bad_samples):
        #     self.manifold_test[bad_samples,:] = np.mean(self.manifold_test[good_samples,:], axis=0)
            
        ## Normalize X_test manifold
        self.manifold_test -= self.centers[1]
        self.manifold_test /= self.scales[1]
        
        ## Transform novel manifold onto familiar manifold
        # R, s = self.transformation
        # self.manifold_test_aligned = (self.manifold_test @ R.T) * s
        R = self.transformation
        self.manifold_test_aligned = self.manifold_test @ R.T
        y_pred = self.decoder.predict(self.manifold_test_aligned)
        
        return y_pred    
    
# %% Spatial decoder with two aligned manifolds (average trials before dimension reduction)
class SpatialNovelDecoder2(object):
    def __init__(self, reduction='none', dim=2, method='knr', k=5, knr_metric='cosine',
                 yrange=(0.,4.), ybin=80, ysigma=1, **param_umap):
        self.reduction = reduction
        self.dim = dim
        self.method = method
        self.k = k  # Number of neighbors in KNeighborsRegressor
        self.knr_metric = knr_metric  # Metric used in KNeighborsRegressor
        self.yrange = yrange
        self.ybin = ybin
        self.ysigma = ysigma
        self.param_umap = param_umap    
        
    def fit(self, X_train, y_train):
        
        nctx = len(X_train)
        label = np.hstack([np.repeat(c, X_train[c].shape[0]) 
                           for c in range(nctx)])  # Two context Fam and Nov
        ## Manifold learn
        if self.reduction == 'none':
            self.X_train_embed = [X_train[c].copy() for c in range(nctx)]
        else:
            if self.reduction == 'pca':
                self.reducer = PCA(n_components=self.dim).fit(np.vstack(X_train))
                X_train_embed = self.reducer.transform(np.vstack(X_train))
            elif self.reduction=='umap':
                self.reducer = umap.UMAP(n_components=self.dim, **self.param_umap).fit(np.vstack(X_train))
                X_train_embed = self.reducer.embedding_.copy()  # Important!! copy array!!
            self.X_train_embed = [X_train_embed[label==c] for c in range(nctx)]

        ## Normalize X_train_embed
        self.X_train_norm = [[] for _ in range(nctx)]
        self.centers = [[] for _ in range(nctx)]
        self.scales = [[] for _ in range(nctx)]
        for c in range(nctx):
            self.X_train_norm[c], self.centers[c], self.scales[c] = \
                normalize_shape(self.X_train_embed[c])
        
        ## Align X_train_norm (rotation/reflection) without scaling
        R, s = orthogonal_procrustes(*self.X_train_norm)
        self.transformation = R
        self.X_train_aligned = [self.X_train_norm[0], self.X_train_norm[1] @ R.T]
        
        ## Train a position decoder based on Fam manifold
        if self.method == 'knr':
            self.decoder = KNeighborsRegressor(
                n_neighbors=self.k, metric=self.knr_metric)  # weight='distance'
        elif self.method == 'gpr':
            self.decoder = GaussianProcessRegressor()
        elif self.method == 'svr':
            self.decoder = SVR(kernel='linear')
        elif self.method == 'pvcorr':
            self.decoder = PVCorr(yrange=self.yrange, ybin=self.ybin, ysigma=self.ysigma)
        elif self.method == 'bayesian':
            self.decoder = NaiveBayes1D(yrange=self.yrange, ybin=self.ybin, ysigma=self.ysigma)
        self.decoder.fit(self.X_train_norm[0], y_train)
        
    def predict(self, X_test, align=True):
        if self.reduction == 'none':
            self.X_test_embed = np.copy(X_test)
        else:
            self.X_test_embed = np.copy(self.reducer.transform(X_test))  # Important!! copy array!!
        
        ## Discard NaN (due to disconnected points ?)
        ind = np.all(np.isfinite(self.X_test_embed), axis=1)
        
        ## Normalize X_test manifold
        size = self.X_test_embed.shape
        self.X_test_norm = np.tile(np.NaN, size)
        self.X_test_norm[ind,:] = normalize_shape(self.X_test_embed[ind,:])[0]  # Normalize by itself
        # self.X_test_norm[ind,:] = (self.X_test_embed[ind,:] - self.centers[1])/self.scales[1]  # Normalize by the size of Nov train 
        
        ## Transform novel manifold onto familiar manifold
        if align:
            R = self.transformation
            self.X_test_aligned = np.tile(np.NaN, size)
            self.X_test_aligned[ind,:] = self.X_test_norm[ind,:] @ R.T
        else:
            self.X_test_aligned = self.X_test_norm
        
        y_pred = np.repeat(np.NaN, size[0])
        y_pred[ind] = self.decoder.predict(self.X_test_aligned[ind,:])
        
        return y_pred
    

    