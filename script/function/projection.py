# -*- coding: utf-8 -*-
"""
Functions displaying 2D/3D neural activities after dimensionality reduction.

@author: Hung-Ling
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm  # For setting up discrete colormap
from matplotlib import markers

# %%
def plot_projection(X, y, fig=None, ax=None, label='', cmap='jet', cbar=True,
                    discrete=False, **kwargs):
    '''
    Parameters
    ----------
    X : numpy 2d array, shape (n_samples,2)
        Projected high dimensional data (to the 2D plane).
    y : numpy 1d array, shape (n_samples,)
        Labels of the data points.
    fig : matplotlib.figure.Figure handle
        Pass the figure where to plot. The default is None (create a new figure)
    ax : matplotlib.axes.Axes handle
        Pass the ax where to plot. The default is None (create a new ax)
    label : str
        Title for the labels.
    cmap : str or matplotlib.colors.ListedColormap
        Colormap used. The default is 'jet'.
    discrete : bool
        Whether to use discrete colors to represent each label number.
        The default is False (label numbers are linearly mapped to the colormap).
    kwargs :
        Additional keyword argument to be passed into the scatter plot function.
    '''
    if isinstance(cmap, str):
        try:
            cmap = eval('plt.cm.'+cmap)
        except AttributeError:
            print(f'Warning: {cmap} is not a valid colormap, use jet instead')
            cmap = plt.cm.jet
    
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = plt.gca()
    
    if discrete:  # Discrete colormap
        vmin = int(np.min(y))
        vmax = int(np.max(y))
        if vmax-vmin+1 > cmap.N:   # Number of colors in the colormap
            print('Warning: number of colors is smaller than number of labels, suggest changing colormap or setting discrete=False')
        bounds = np.linspace(vmin-0.5, vmax+0.5, vmax-vmin+2)  # Data boundaries used to map into different colors
        norm = BoundaryNorm(bounds, vmax-vmin+1)  # Generate a colormap index based on discrete intervals
                                                 # vmax-vmin+1 is the number of colors to be used
        cax = ax.scatter(X[:,0], X[:,1], c=y, cmap=cmap, norm=norm, **kwargs)
        if cbar:
            fig.colorbar(cax, ax=ax, ticks=np.arange(vmin,vmax+1), label=label)
    else:  # Continuous colormap
        cax = ax.scatter(X[:,0], X[:,1], c=y, cmap=cmap, **kwargs)
        if cbar:
            fig.colorbar(cax, ax=ax, label=label)
    
    ax.set_aspect('equal', 'datalim')
    fig.tight_layout()

# %%
def plot_projection_bis(Xs, ys, fig=None, ax=None, markerlist=None, label='', cmap='jet', **kwargs):
    
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = plt.gca()
    if markerlist is None:
        markerlist = markers.MarkerStyle().filled_markers
    for i, (X, y) in enumerate(zip(Xs, ys)):
        ii = i % len(markerlist)
        cax = ax.scatter(X[:,0], X[:,1], c=y, marker=markerlist[ii], cmap=cmap, **kwargs)
    fig.colorbar(cax, ax=ax, label=label, location='bottom')
        
# %%
def plot_line_boundary(clf, xrange, ax=None, **kwargs):
    '''Plot the separating hyperplane of a binary classifier.
    '''
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(*xrange)
    yy = a * xx - (clf.intercept_[0]) / w[1]
    if ax is None:
        ax = plt.gca()
    ax.plot(xx, yy, **kwargs)
    
# %% 
def plot_decision_region(clf, xrange, yrange, ax=None, **kwargs):
    '''Plot the linear decision regions of a binary classifier.
    '''
    xx, yy = np.meshgrid(np.linspace(*xrange,1000), np.linspace(*yrange,1000))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    if ax is None:
        ax = plt.gca()
    # ax.contourf(xx, yy, Z, **kwargs)
    ax.pcolormesh(xx, yy, Z, shading='auto', **kwargs)
    
# %%
def plot_projection_3d(X, y, fig=None, ax=None, label='', cmap='jet', cbar=True, 
                       discrete=False, **kwargs):
    '''3d version of plot_projection
    '''
    if isinstance(cmap, str):
        try:
            cmap = eval('plt.cm.'+cmap)
        except AttributeError:
            print(f'Warning: {cmap} is not a valid colormap, use jet instead')
            cmap = plt.cm.jet
    
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = plt.gcf().add_subplot(projection='3d')
    
    if discrete:  # Discrete colormap
        vmin = int(np.min(y))
        vmax = int(np.max(y))
        if vmax-vmin+1 > cmap.N:   # Number of colors in the colormap
            print('Warning: number of colors is smaller than number of labels, suggest changing colormap or setting discrete=False')
        bounds = np.linspace(vmin-0.5, vmax+0.5, vmax-vmin+2)  # Data boundaries used to map into different colors
        norm = BoundaryNorm(bounds, vmax-vmin+1)  # Generate a colormap index based on discrete intervals
                                                 # vmax-vmin+1 is the number of colors to be used
        cax = ax.scatter(X[:,0], X[:,1], X[:,2], c=y, cmap=cmap, norm=norm, **kwargs)
        if cbar:
            fig.colorbar(cax, ax=ax, ticks=np.arange(vmin,vmax+1), label=label)
    else:  # Continuous colormap
        cax = ax.scatter(X[:,0], X[:,1], X[:,2], c=y, cmap=cmap, **kwargs)
        if cbar:
            fig.colorbar(cax, ax=ax, label=label)
    
    fig.tight_layout()

# %%
def display_landmark(landmarks, yrange=(0,4), ybin=76, 
                     colors=['dimgray','lightgray'], cbar=True):
    
    y_edges = np.linspace(*yrange, ybin+1)
    y_centers = (y_edges[:-1] + y_edges[1:])/2
    if isinstance(landmarks, np.ndarray):
        landmarks = [landmarks]
    
    if landmarks[0].shape[1] == 2:
        fig, ax = plt.subplots()
        for c, landmark in enumerate(landmarks):
            cax = ax.scatter(landmark[:,0], landmark[:,1], c=y_centers,
                             cmap='jet', s=20, alpha=0.8)
            ax.plot(landmark[:,0], landmark[:,1], c=colors[c])
        if cbar:    
            fig.colorbar(cax, ax=ax, label='Position')    
    
    elif landmarks[0].shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for c, landmark in enumerate(landmarks):
            cax = ax.scatter(landmark[:,0], landmark[:,1], landmark[:,2], c=y_centers,
                             cmap='jet', s=20, alpha=0.8)
            ax.plot3D(landmark[:,0], landmark[:,1], landmark[:,2], c=colors[c])
        ax.grid(False)
        if cbar:
            fig.colorbar(cax, ax=ax, label='Position')
    else:
        print('Data dimension is not 2 or 3, unable to visualize')
        
        