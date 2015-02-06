#Written by Sebastian Otalora Based on Ding's Paper: Convex and Semi-Nonnegative Matrix Factorizations
#This function finds the Matrix Factors W and H of K = KWH Where:
#     K is a kernel matrix of size NxN and K has a positive part denoted XXP and a negative part denoted XXN
#     W0 and H0 are initializations for the matrix factors



import numpy as np
from numba import double, jit

#if len(sys.argv) < 5:
#  print 'Use: nmfCoding.py Initial H, Inital W, Positive part of kernel, Negative part of kernel, iterations, '
#  sys.exit()

@jit(argtypes=[double[:,:], double[:,:],double[:,:],double[:,:]])
def CNMF(H0, W0, XXP, XXN):
    
    iterations = 100

    
    H = H0
    W = W0
    #err = np.zeros(iterations)
    for i in range(1,iterations):
        print 'Iteration: ', i
        Ht = np.multiply(H,np.sqrt((XXP*W+H*W.transpose()*XXN*W) / (XXN*W+H*W.transpose()*XXP*W)))

        Wt = np.multiply(W,np.sqrt((XXP*H+XXN*W*(H.transpose()*H)) / (XXN*H+XXP*W*(H.transpose()*H ))))
        H = Ht
        W = Wt
    return [W,H.transpose()]

def CNMF_W_FIXED(H0, W, XXP, XXN,YYP, YYN):
    iterations = 150
    H = H0

    for i in range(1,iterations):
        print 'W-Fixed Iteration: ', i
        Ht = np.multiply(H,np.sqrt((XXP*W+H*W.transpose()*YYN*W) / (XXN*W+H*W.transpose()*YYP*W)))
        H = Ht;

    return H.transpose()

