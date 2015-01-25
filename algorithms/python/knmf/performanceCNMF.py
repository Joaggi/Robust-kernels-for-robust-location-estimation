from hdf5 import *
import mfOnlineJointLearning as mf
import CNMF as cnmf
import scipy.io as sio
import numpy as np
import histersection as hi
#import mindlabUtilities as mlutilities
from numba import jit, void, double

datasetName = 'Corel5k.h5' 
h5_file = h5py.File(datasetName, 'r')
dataset = h5_file['/']
T  = np.matrix(dataset['/textualtrain'])
Tp = np.matrix(dataset['/textualtest'])
V  = np.matrix(dataset['/visualtrain']) 
#V = V + 0.79743999 #min of the absolute values for yeast dataset
Vp = np.matrix(dataset['/visualtest']) 
#Vp = Vp + 0.66061997
numberOfKNNLabels = 20
#Build of the linear kernel
TT  = T*np.transpose(T)
Ktn = np.divide(TT, np.sqrt(np.dot(TT.diagonal().transpose(),TT.diagonal()) )) #Normalization
#VV = hi.histersection(V,V)

# If we are loading the kernel from a plaintxt file
#VV = np.matrix(np.loadtxt('visualKernel.txt', delimiter=',', unpack=True))
VV = hi.histersection(V,V)
Ktv = np.divide(VV, np.sqrt(np.dot(VV.diagonal().transpose(),VV.diagonal()) )) #Normalization
#KFeatn = KFeat./(sqrt(diag(KFeat)*diag(KFeat)'));

#Weighted Tensor Kernel
alpha = 0.7
KT = np.multiply(alpha, np.log(Ktn))  + np.multiply((1-alpha),np.log(Ktv))
KT = np.exp(KT);

# UNTIL HERE IS TESTED THE OUTPUT AND CORRESPONDS TO THE MATLAB IMPLEMENTATION
r   = 40 #number of latent topics
ndc = V.shape[0] 

XXP = np.divide(np.abs(KT) + KT,2);
XXN = np.divide((np.abs(KT) - KT),2);
# For deterministic testing: 
#s = (4500,40)
#WI  = np.matrix(np.ones(s))
HI  = np.matrix(np.random.random((r,ndc)))  
WI  = np.matrix(np.random.random((ndc,r))) 
norm = 0;

# Calculating latent space basis
WH = cnmf.CNMF(HI.transpose(), WI, XXP, XXN)
W = WH[0]
H = WH[1]

# Calculating test images latent representation
ntt  = T.shape[1]
Vpc  = np.multiply(1.00/(ntt) ,Vp.transpose())
Vp  = Vp.transpose()
KQv  = (hi.histersection(Vpc,V.transpose())).transpose()

vpt = np.sqrt((Vp.transpose()*Vp).diagonal().transpose())
sqf = np.sqrt(VV.diagonal()) 

KQvn = np.divide(KQv, vpt*sqf)


N = Vp.shape[1];
iter = 200;

YYP = XXP
YYN = XXN

XXP = np.divide(np.abs(KQvn) + KQvn,2);
XXN = np.divide(np.abs(KQvn) - KQvn,2);
HI  = np.matrix(np.random.random((r,N)))  
U   = cnmf.CNMF_W_FIXED(HI.transpose(),W, XXP, XXN, YYP, YYN);


#The ranking is the convex projection of the queries latent representation and the original samples 
R = KT*W*U
#What about a ranking based on the histersection measure?
#R = histersection(KT,WU)

sio.savemat('rankingCorelPython.mat',{'R':R})

#Obtain the most frequent labels from its KNN neighbours
#Tn = mlutils.rankingLabels(R,T,numberOfKNNLabels)

#Obtain the performance measures for the predicted labels
#[params] = evalLabeling(Tp',Tn',params);
#       results4 = [params,results4];
