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

datasetPath = 'Corel5k.h5'

#Load Dataset Original Features
[T,Tp,V,Vp] = dl.loadH5Dataset(datasetPath)

#Build Cosine similarity kernel for text features
Ktext = skl.metrics.pairwise_kernels(T,T,metric='cosine')
Ktext = Ktext +10


#Normalize Visual Kernel
Kvisual = hi.histersection(V.T,V.T)

Ktv = np.divide(Kvisual, np.sqrt(np.dot(Kvisual.diagonal().transpose(),Kvisual.diagonal()) )) #Normalization


#Visual Kernel smoothing
Kvisual = Kvisual+0.00001;

#Weighted Tensor Kernel
alpha = 0.7
KT = np.multiply(alpha, np.log(Ktext))  + np.multiply((1-alpha),np.log(Kvisual))
KT = np.exp(KT)

[XXP, XXN] = dl.loadPositiveNegative(KT)

r = 40
ndc = Ktext.shape[0]
HI  = np.matrix(np.random.random((r,ndc)))  
WI  = np.matrix(np.random.random((ndc,r))) 

[W, H] = cnmf.CNMF(HI.transpose(), WI, XXP, XXN)


#Computing test-training visual kernel

KQv  = (hi.histersection(Vp.T,V.transpose())).transpose()
KQv = KQv + 0.00001


uniform = 1.0/T.shape[1]
unif = np.multiply(uniform,np.ones((500,374)))
KQt = skl.metrics.pairwise_kernels(unif,T,metric='cosine')

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

recKt = np.divide(R, KQv)


sio.savemat('rankingCorelPython.mat',{'R':R})
sio.savemat('rankingCorelPython.mat',{'recKt':recKt})












