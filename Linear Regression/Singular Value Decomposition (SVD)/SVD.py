# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 17:06:37 2017

@author: Stagiaire
"""
import numpy as np
from numpy.linalg import svd
 
movieRatings = [
    [2, 5, 3],
    [1, 2, 1],
    [4, 1, 1],
    [3, 5, 2],
    [5, 3, 1],
    [4, 5, 5],
    [2, 4, 2],
    [2, 2, 5],
]
 
U, singularValues, V = svd(movieRatings)

Sigma = np.vstack([
    np.diag(singularValues),
    np.zeros((5, 3)),
])
 
m= np.round(movieRatings - np.dot(U, np.dot(Sigma, V)), decimals=10)

import numpy as np
from numpy.linalg import norm
 
from random import normalvariate
from math import sqrt
 
def randomUnitVector(n):
    unnormalized = [normalvariate(0, 1) for _ in range(n)]
    theNorm = sqrt(sum(x * x for x in unnormalized))
    return [x / theNorm for x in unnormalized]
 
def svd_1d(A, epsilon=1e-10):
    ''' The one-dimensional SVD '''
 
    n, m = A.shape
    x = randomUnitVector(m)
    lastV = None
    currentV = x
    B = np.dot(A.T, A)
 
    iterations = 0
    while True:
        iterations += 1
        lastV = currentV
        currentV = np.dot(B, lastV)
        currentV = currentV / norm(currentV)
 
        if abs(np.dot(currentV, lastV)) > 1 - epsilon:
            print("converged in {} iterations!".format(iterations))
            return currentV
        
if __name__ == "__main__":
    movieRatings = np.array([
        [2, 5, 3],
        [1, 2, 1],
        [4, 1, 1],
        [3, 5, 2],
        [5, 3, 1],
        [4, 5, 5],
        [2, 4, 2],
        [2, 2, 5],
    ], dtype='float64')
 
    print(svd_1d(movieRatings))
def svd(A, epsilon=1e-10):
    n, m = A.shape
    svdSoFar = []
 
    for i in range(m):
        matrixFor1D = A.copy()
 
        for singularValue, u, v in svdSoFar[:i]:
            matrixFor1D -= singularValue * np.outer(u, v)
 
        v = svd_1d(matrixFor1D, epsilon=epsilon)  # next singular vector
        u_unnormalized = np.dot(A, v)
        sigma = norm(u_unnormalized)  # next singular value
        u = u_unnormalized / sigma
 
        svdSoFar.append((sigma, u, v))
 
    # transform it into matrices of the right shape
    singularValues, us, vs = [np.array(x) for x in zip(*svdSoFar)]
 
    return singularValues, us.T, vs

theSVD = svd(movieRatings)