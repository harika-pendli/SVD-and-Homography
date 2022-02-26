# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 11:40:09 2022

@author: pendl
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def computeSVD(mat):
    
    A=np.array(mat)
    m , n =A.shape
    #print(A.shape)
    AAt=np.dot(A,A.T)
    AtA=np.dot(A.T,A) 
            
    w, U = np.linalg.eig(AAt)
    idx = np.flip(np.argsort(w))
    eval_AAt = w[idx]
    U = U[:, idx]
        
    p = min(m,n)
    sigma=np.zeros((m,n))
    for i in range(p):
        sigma[i,i]= np.abs(np.sqrt(eval_AAt[i]))
    
    w_2, V = np.linalg.eig(AtA)
    idx = np.flip(np.argsort(w_2))
    eval_AtA = w_2[idx]
    V= V[:,idx]
    return U,sigma,V
    
    
    
def computeHomography(ps1, ps2):

    if (len(ps1) < 4) or (len(ps2) < 4):
        print("Need atleast four points to compute SVD.")
        return 0

    x = ps1[:, 0]
    y = ps1[:, 1]
    xp = ps2[:, 0]
    yp = ps2[:,1]

    nrows= 8
    
    
    A = []
    for i in range(int(nrows/2)):
        row1 = np.array([-x[i], -y[i], -1, 0, 0, 0, x[i]*xp[i], y[i]*xp[i], xp[i]])
        A.append(row1)
        row2 = np.array([0, 0, 0, -x[i], -y[i], -1, x[i]*yp[i], y[i]*yp[i], yp[i]])
        A.append(row2)

    A = np.array(A)
    print("Computing Homography matrix for ")
    print(A)
    U, E, V = computeSVD(A)
    H_vertical = V[:, V.shape[1] - 1]
    H = H_vertical.reshape([3,3])
    H = H / H[2,2]
    return H

pointset1 = np.array([[5, 5], [150, 5], [150, 150], [5, 150]])
pointset2 = np.array([[100, 100], [200, 80], [220, 80], [100, 200]])

computeHomography(pointset1, pointset2)