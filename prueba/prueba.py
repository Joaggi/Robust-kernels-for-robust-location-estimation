# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 22:07:38 2014

@author: Alejandro
"""
import os, sys
#lib_path = os.path.abspath('../../../Algorithms/python')
lib_path = os.path.abspath('G:/Dropbox/Universidad/Machine Learning')
sys.path.append(lib_path)

import numpy as np
import scipy.io as sio
import Algorithms.Python.normalizeByRange as normalizeByRange
import Algorithms.Python.gaussianKernel as gaussianKernel
import Algorithms.Python.clustering.kernel_k_means as kernel_k_means
import Algorithms.Python.clustering.accuracy as accuracy
import Algorithms.Python.Metrics as metrics 
import Algorithms.Python.KernelKMeans as KernelKMeans 

#fileData = 'G:/Dropbox/Universidad/Machine Learning/Robustes/AR/AR.npz'
fileData = 'G:/Dropbox/Universidad/Machine Learning/Robustes/jaffe/jaffe.npz'
epocs = sio.loadmat('G:/Dropbox/Universidad/Machine Learning/Robustes/ATT/parameters.mat')['epocs']
data = np.load(fileData)
X = data['data']
labelsReal = data['labels']

sigma = 2**3
nClusters = 10


X = normalizeByRange.normalizeByRange(X,0)
X= np.transpose(X)
K =  gaussianKernel.gaussianKernel(X,sigma=sigma)
H,labelsPred,t =  kernel_k_means.KKmeans(K,nClusters,1000)
print labelsPred
print accuracy.accuracy(labelsReal,labelsPred)
print metrics.calculate_clusteringAccuracy(labelsReal,labelsPred)




model = KernelKMeans.KernelKMeans(n_clusters=nClusters, 
                         kernel="rbf", gamma=1./sigma,
                         verbose=1)
print model
model.fit(np.transpose(X))
labelsPred = model.labels_

print accuracy.accuracy(labelsReal,labelsPred)
print metrics.calculate_clusteringAccuracy(labelsReal,labelsPred)

