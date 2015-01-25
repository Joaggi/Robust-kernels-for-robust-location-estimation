# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 17:31:02 2014

@author: Alejandro
"""

#!/usr/bin/env python
# a bar plot with errorbars
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

accuracy = np.zeros(7)
nmi = np.zeros(7)
purity = np.zeros(7)
location = np.zeros(7)

location[0] = sio.loadmat('KMeans/results')['clusteringAccuracyMeanVec'].argmax()
location[1] = sio.loadmat('KernelKMeans/results')['clusteringAccuracyMeanVec'].argmax()
location[2] = sio.loadmat('NNMF/results')['clusteringAccuracyMeanVec'].argmax()
location[3] = sio.loadmat('RMNMF/results')['clusteringAccuracyMeanVec'].argmax()
location[4] = sio.loadmat('Kernelconvexnmf/results')['clusteringAccuracyMeanVec'].argmax()
location[5] = sio.loadmat('Kernelseminmfnnls/results')['clusteringAccuracyMeanVec'].argmax()
location[6] = sio.loadmat('Kernelseminmfrule/results')['clusteringAccuracyMeanVec'].argmax()

accuracy[0] = max(sio.loadmat('KMeans/results')['clusteringAccuracyMeanVec'])
accuracy[1] = max(sio.loadmat('KernelKMeans/results')['clusteringAccuracyMeanVec'])
accuracy[2] = max(sio.loadmat('NNMF/results')['clusteringAccuracyMeanVec'])
accuracy[3] = real(max(sio.loadmat('RMNMF/results')['clusteringAccuracyMeanVec']))
accuracy[4] = max(sio.loadmat('Kernelconvexnmf/results')['clusteringAccuracyMeanVec'])
accuracy[5] = max(sio.loadmat('Kernelseminmfnnls/results')['clusteringAccuracyMeanVec'])
accuracy[6] = max(sio.loadmat('Kernelseminmfrule/results')['clusteringAccuracyMeanVec'])

nmi[0] = sio.loadmat('KMeans/results')['nmiMeanVec'][location[0]]
nmi[1] = sio.loadmat('KernelKMeans/results')['nmiMeanVec'][location[1]]
nmi[2] = sio.loadmat('NNMF/results')['nmiMeanVec'][location[2]]
nmi[3] = real(sio.loadmat('RMNMF/results')['nmiMeanVec'][location[3]])
nmi[4] = sio.loadmat('Kernelconvexnmf/results')['nmiMeanVec'][location[4]]
nmi[5] = real(sio.loadmat('Kernelseminmfnnls/results')['nmiMeanVec'][location[5]])
nmi[6] = sio.loadmat('Kernelseminmfrule/results')['nmiMeanVec'][location[6]]

purity[0] = sio.loadmat('KMeans/results')['purityMeanVec'][location[0]]
purity[1] = sio.loadmat('KernelKMeans/results')['purityMeanVec'][location[1]]
purity[2] = sio.loadmat('NNMF/results')['purityMeanVec'][location[2]]
purity[3] = sio.loadmat('RMNMF/results')['purityMeanVec'][location[3]]
purity[4] = sio.loadmat('Kernelconvexnmf/results')['purityMeanVec'][location[4]]
purity[5] = sio.loadmat('Kernelseminmfnnls/results')['purityMeanVec'][location[5]]
purity[6] = sio.loadmat('Kernelseminmfrule/results')['purityMeanVec'][location[6]]

KMeans = [accuracy[0] , nmi[0] , purity[0] , 0]
KernelKMeans = [accuracy[1] , nmi[1] , purity[1] , 0]
NMF = [accuracy[2] , nmi[2] , purity[2] , 0]
RMNMF = [accuracy[3] , nmi[3] , purity[3] , 0]
Kernelconvexnmf = [accuracy[4] , nmi[4] , purity[4] , 0]
Kernelseminmfnnls = [accuracy[5] , nmi[5] , purity[5] , 0]
Kernelseminmfrule = [accuracy[6] , nmi[6] , purity[6] , 0]

N = 4

ind = np.arange(N)  # the x locations for the groups
width = 0.10       # the width of the bars

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
ax.set_title('BalanceScale')
ax.set_xticks(ind+4*width)
ax.set_xticklabels( ('Clust Accura', 'NMI', 'Purity') )

ax.legend( (rects1[0], rects2[0],rects3[0],rects4[0],rects5[0],rects6[0],rects7[0]), ('KMeans', 'KernelKMeans' , 'NMF' , 'RMNMF' ,  'KConvexnmf', 'KSemNmfNnls' , 'KSeminmfrule' ) ,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
                ha='center', va='bottom')

#autolabel(rects1)
#autolabel(rects2)

plt.show()