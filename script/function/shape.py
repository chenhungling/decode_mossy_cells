"""
Align shape B to A using the Procrustes transformation.

Adapted from C:\ProgramData\Anaconda3\envs\neurokit\Lib\site-packages\scipy\spatial\_procrustes.py

@author: Hung-Ling
"""
import numpy as np
from scipy.linalg import orthogonal_procrustes

def procrustes(A, B):
    '''
    Procrustes analysis, a similarity test for two data sets `A` and `B`.

    Each input matrix is a set of points or vectors (the rows of the matrix).
    The dimension of the space is the number of columns of each matrix.
    Given two identically sized matrices, procrustes standardizes both such that:
    
    - Both sets of points are centered around the origin.
    - `A` and `B` will be scaled such that the Frobenius norm is unity: Trace(X @ X.T) = 1
    
    Procrustes then applies the optimal transform to the second matrix `B`
    (including scaling/dilation, rotations, and reflections) to minimize the 
    sum of the squares of the pointwise differences: sum((A - B)**2)
        
    Parameters
    ----------
    A : numpy 2d array, shape (n_points, dimension)
        The reference data, after it is standardised.
    B : numpy 2d array, shape (n_points, dimension)
        The data that will be transformed to fit the pattern in `A`.
        Must be the same shape as `A`
    
    Returns
    -------
    (A2, B2) : (numpy 2d array, numpy 2d array)
        Standardized version of `A`, and `B` that best fits `A` (centered,
        but not necessarily: Trace(B @ B.T) = 1)
    parameters : dict, other useful returned values
        'transformation': (R, s), (array, float). The orthogonal matrix and 
            scaling factor for transforming B to A
        'centers': (numpy 1d array, numpy 1d array). The centers of the original input data.
        'scales': (float, float). The Frobenius norms of the original input data.
        'disparity': float. 
    '''
    A1 = np.array(A, dtype=float, copy=True)
    B1 = np.array(B, dtype=float, copy=True)
    
    ## Center shapes to the origin
    centerA = np.mean(A, axis=0)
    centerB = np.mean(B, axis=0)
    A1 = A1 - centerA[np.newaxis,:]
    B1 = B1 - centerB[np.newaxis,:]
    
    # Change scaling (in rows) such that trace(X @ X.T) = 1
    normA = np.linalg.norm(A1)  # Same as np.sqrt(np.sum(A1**2))
    normB = np.linalg.norm(B1)  # Same as np.sqrt(np.sum(B1**2))
    A1 = A1/normA
    B1 = B1/normB
    
    R, s = orthogonal_procrustes(A1, B1)
    B2 = (B1 @ R.T) * s
    disparity = np.sum((A1 - B2)**2)
    
    parameters = {
        'transformation': [R, s],
        'centers': [centerA, centerB],
        'scales': [normA, normB],
        'disparity': disparity
        }
    
    return (A1, B2), parameters


