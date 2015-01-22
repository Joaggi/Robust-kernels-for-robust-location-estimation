# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 17:31:02 2014

@author: Alejandro
"""

#!/usr/bin/env python
# a bar plot with errorbars

import os, sys
lib_path = os.path.abspath('G:/Dropbox/Universidad/Machine Learning')
sys.path.append(lib_path)
    
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

def bar_accuracy(database, namedatabase):
    accuracy = np.zeros(7)
    nmi = np.zeros(7)
    purity = np.zeros(7)
    location = np.zeros(7)
    
    location[0] = np.nanargmax(sio.loadmat(database+'KMeans/results')['clusteringAccuracyMeanVec'])
    location[1] = np.nanargmax(sio.loadmat(database+'KernelKMeans/results')['clusteringAccuracyMeanVec'])
    location[2] = np.nanargmax(sio.loadmat(database+'NNMF/results')['clusteringAccuracyMeanVec'])
    location[3] = np.nanargmax(sio.loadmat(database+'RMNMF/results')['clusteringAccuracyMeanVec'])
    location[4] = np.nanargmax(sio.loadmat(database+'Kernelconvexnmf/results')['clusteringAccuracyMeanVec'])
    location[5] = np.nanargmax(sio.loadmat(database+'Kernelseminmfnnls/results')['clusteringAccuracyMeanVec'])
    location[6] = np.nanargmax(sio.loadmat(database+'Kernelseminmfrule/results')['clusteringAccuracyMeanVec'])
    
    accuracy[0] = np.nanmax(sio.loadmat(database+'KMeans/results')['clusteringAccuracyMeanVec'])
    accuracy[1] = np.nanmax(sio.loadmat(database+'KernelKMeans/results')['clusteringAccuracyMeanVec'])
    accuracy[2] = np.nanmax(sio.loadmat(database+'NNMF/results')['clusteringAccuracyMeanVec'])
    accuracy[3] = (np.nanmax(sio.loadmat(database+'RMNMF/results')['clusteringAccuracyMeanVec']))
    accuracy[4] = np.nanmax(sio.loadmat(database+'Kernelconvexnmf/results')['clusteringAccuracyMeanVec'])
    accuracy[5] = np.nanmax(sio.loadmat(database+'Kernelseminmfnnls/results')['clusteringAccuracyMeanVec'])
    accuracy[6] = np.nanmax(sio.loadmat(database+'Kernelseminmfrule/results')['clusteringAccuracyMeanVec'])
    
    nmi[0] = sio.loadmat(database+'KMeans/results')['nmiMeanVec'][location[0]]
    nmi[1] = sio.loadmat(database+'KernelKMeans/results')['nmiMeanVec'][location[1]]
    nmi[2] = sio.loadmat(database+'NNMF/results')['nmiMeanVec'][location[2]]
    nmi[3] = np.real(sio.loadmat(database+'RMNMF/results')['nmiMeanVec'][location[3]])
    nmi[4] = sio.loadmat(database+'Kernelconvexnmf/results')['nmiMeanVec'][location[4]]
    nmi[5] = sio.loadmat(database+'Kernelseminmfnnls/results')['nmiMeanVec'][location[5]]
    nmi[6] = sio.loadmat(database+'Kernelseminmfrule/results')['nmiMeanVec'][location[6]]
    
    purity[0] = sio.loadmat(database+'KMeans/results')['purityMeanVec'][location[0]]
    purity[1] = sio.loadmat(database+'KernelKMeans/results')['purityMeanVec'][location[1]]
    purity[2] = sio.loadmat(database+'NNMF/results')['purityMeanVec'][location[2]]
    purity[3] = sio.loadmat(database+'RMNMF/results')['purityMeanVec'][location[3]]
    purity[4] = sio.loadmat(database+'Kernelconvexnmf/results')['purityMeanVec'][location[4]]
    purity[5] = sio.loadmat(database+'Kernelseminmfnnls/results')['purityMeanVec'][location[5]]
    purity[6] = sio.loadmat(database+'Kernelseminmfrule/results')['purityMeanVec'][location[6]]
    
    KMeans = [accuracy[0] , nmi[0] , purity[0] , 0,0]
    KernelKMeans = [accuracy[1] , nmi[1] , purity[1] , 0,0]
    NMF = [accuracy[2] , nmi[2] , purity[2] , 0,0]
    RMNMF = [accuracy[3] , nmi[3] , purity[3] , 0,0]
    Kernelconvexnmf = [accuracy[4] , nmi[4] , purity[4] , 0,0]
    Kernelseminmfnnls = [accuracy[5] , nmi[5] , purity[5] , 0,0]
    Kernelseminmfrule = [accuracy[6] , nmi[6] , purity[6] , 0,0]
    
    N = 5
    
    ind = np.arange(N)  # the x locations for the groups
    width = 0.12       # the width of the bars
    
    fig, ax = plt.subplots()
    color = ['b','g','#ba4545','y','#887abb','#5a9c7e','#ab9054']
    rects1 = ax.bar(ind, KMeans, width, color=color[0])
    rects2 = ax.bar(ind+width, KernelKMeans, width, color=color[1])
    rects3 = ax.bar(ind+2*width, NMF, width, color=color[2])
    rects4 = ax.bar(ind+3*width, RMNMF, width, color=color[3])
    rects5 = ax.bar(ind+4*width, Kernelconvexnmf, width, color=color[4])
    rects6 = ax.bar(ind+5*width, Kernelseminmfnnls, width, color=color[5])
    rects7 = ax.bar(ind+6*width, Kernelseminmfrule, width, color=color[6])
    
    # add some
    #ax.set_ylabel('Scores')
    #ax.set_title(namedatabase,fontsize=18)
    ax.set_xticks(ind+4*width)
    ax.set_xticklabels( ('ACC', 'NMI', 'Purity') ,fontsize=18)
    ax.set_ylabel( 'Clustering Measures' ,fontsize=18)
    
    ax.set_autoscaley_on(False)
    #ax.set_ylim([0.5,0.92])
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    ax.legend( (rects1[0], rects2[0],rects3[0],rects4[0],rects5[0],rects6[0],rects7[0]), ('KMeans', 'KernelKMeans' , 'NMF' , 'RMNMF' ,  'KConvexnmf', 'KSemNmfNnls' , 'KSeminmfrule' ) ,bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)
    
    plt.subplots_adjust(left=-0.1, bottom=None, right=None, top=None,
                    wspace=None, hspace=None)
    plt.savefig(namedatabase, dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        frameon=None)
    

    
#autolabel(rects1)
#autolabel(rects2)