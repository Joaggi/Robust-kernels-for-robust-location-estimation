from hdf5 import *
import mfOnlineJointLearning as mf
import CNMF as cnmf
import scipy.io as sio
from sklearn.svm import SVC
import sklearn as skl
import numpy as np
import histersection as hi
import datasetload as dl
import graphKernels as gk
from numba import jit, void, double
import generation_GMM as gm

#Generation of synthetic dataset:
[V,T] = gm.generateGMM(1000,2,.5,[2,13],False)
[Vp,Tp] = gm.generateGMM(100,2,.5,[2,13],False)

V = np.matrix(V)
V = V - np.min(V)
Vp = np.matrix(Vp)
Vp = V - np.min(Vp)
T = np.matrix(T)
T = T - np.min(T)
Tp = np.matrix(Tp)
Tp = Tp - np.min(Tp)

Ktext = skl.metrics.pairwise_kernels(T,T,metric='cosine')
Kvisual = hi.histersection(V.T,V.T)
Ktv = np.divide(Kvisual, np.sqrt(np.dot(Kvisual.diagonal().transpose(),Kvisual.diagonal()) )) #Normalization

alphas = [0.2,0.4,0.6,0.8,0.9]
factors = [10, 30, 70, 90, 200, 500, 1000]
#Visual Kernel smoothing
Kvisual = Kvisual+0.00001;
for alpha in alphas:
    #Weighted Tensor Kernel
    #alpha = 0.7
    KT = np.multiply(alpha, np.log(Ktext))  + np.multiply((1-alpha),np.log(Kvisual))
    KT = np.exp(KT)

    [XXP, XXN] = dl.loadPositiveNegative(KT)
    for r in factors:
    #r = 40
        ndc = Ktext.shape[0]
        HI  = np.matrix(np.random.random((r,ndc)))  
        WI  = np.matrix(np.random.random((ndc,r))) 

        [W, H] = cnmf.CNMF(HI.transpose(), WI, XXP, XXN)


        #Computing test-training visual kernel

        KQv  = (hi.histersection(Vp.T,V.transpose())).transpose()
        KQv = KQv + 0.00001


        uniform = 1.0/T.shape[1]
        unif = np.multiply(uniform,np.ones((200,2)))
        KQt = skl.metrics.pairwise_kernels(unif,T,metric='cosine')
        realKQt = skl.metrics.pairwise_kernels(Tp,T,metric='cosine')
        rKQT = np.multiply(alpha, np.log(KQt)) + np.multiply((1-alpha), np.log(KQv))
        rKQT = np.exp(rKQT)

        #True test textual kernel: rKQt = skl.metrics.pairwise_kernels(Tp,T,metric='cosine'
        N = Vp.shape[0]
        iter = 200

        [YYP, YYN] = dl.loadPositiveNegative(KT)

        [XXP, XXN] = dl.loadPositiveNegative(rKQT)

        HI  = np.matrix(np.random.random((r,N)))
        U   = cnmf.CNMF_W_FIXED(HI.transpose(),W, XXP, XXN, YYP, YYN)

        # Recovering text Kernel
        R = KT*W*U

        recKt = np.divide(R, KQv.transpose())
        errorS = str(np.linalg.norm(realKQt - recKt))
        print 'The error of Ktext rec is: ' + errorS + 'con alpha = ' + str(alpha) + ', y '+ str(r) + ' factores latentes'
        
#sio.savemat('rankingSynthetic.mat',{'R':R})
#sio.savemat('TpreconstructionSynthetic.mat',{'recKt':recKt})



