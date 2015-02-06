"""
Created on Tue Aug 26 12:06:01 2014

@author: Esteban
"""


import numpy
import numpy.linalg as la
import math
from numba import jit
from time import time as clock
from sklearn.metrics import pairwise_kernels

@jit('pyobject(double[:,:],int32,int32)')
def KKmeans(K, latentTopics, epochs):
    t0 = clock()
    H,cluster = train(K, epochs, latentTopics)
    t1 = clock()
    return (H,cluster, t1-t0)

@jit('double[:,:](double[:,:],int32,int32)')
def train(K, epochs, latentTopics):
    dataSetSize = K.shape[0]
    H = numpy.transpose(initH(K, latentTopics, dataSetSize))
    for i in xrange(epochs):
        distToCentroids = -2*numpy.dot(H, K) \
            + numpy.diag((numpy.dot(H, K)*H))[:, numpy.newaxis]
        cluster = numpy.argmin(distToCentroids, axis = 0)
        minDist = numpy.min(distToCentroids, axis = 0)
        ix = numpy.argsort(minDist)
        aux_id = 1
        for j in xrange(latentTopics):
            if numpy.sum(numpy.where(cluster==j))==0:
                cluster[ix[aux_id]] = 1
                aux_id = aux_id+1
        H = numpy.zeros((latentTopics,dataSetSize))
        H[cluster,numpy.arange(dataSetSize)] = 1
        H = H*(1/H.sum(axis = 1))[:, numpy.newaxis]
    return H,cluster

@jit('double[:,:](double[:,:],int32,int32)')
def initH(K, latentTopics, dataSetSize):
    pos = numpy.random.randint(dataSetSize,size=latentTopics)
    distance = numpy.dot(numpy.diagonal(K[pos,:][:,pos])[:,numpy.newaxis],numpy.ones((1,dataSetSize)))+ \
           numpy.dot(numpy.ones((latentTopics,1)),numpy.diag(K)[numpy.newaxis]) -\
           2 * K[pos,:]
    idx = numpy.argmin(distance,axis=0)
    H = numpy.zeros((dataSetSize,latentTopics))
    H[numpy.arange(dataSetSize),idx] = 1
    return H*(1/H.sum(axis = 1))[:, numpy.newaxis]


def predictH(K, H, KX):
    distToCentroids = -2*numpy.dot(H, KX) \
        + (numpy.dot(H, K)*H).sum(axis = 1)[:, numpy.newaxis]
    return numpy.argmin(distToCentroids, axis = 0)
