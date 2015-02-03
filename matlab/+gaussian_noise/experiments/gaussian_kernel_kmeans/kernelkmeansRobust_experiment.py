# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 16:55:56 2014

@author: Alejandro
"""
import os, sys
lib_path = os.path.abspath('G:/Dropbox/Universidad/Machine Learning')
sys.path.append(lib_path)
import Algorithms.Python.KernelKMeans as KernelKMeans
import numpy as np
import Algorithms.Python.Metrics as Metrics
import scipy.io as sio

def kernelkmeans(fileData, epocs ,normalized_axis = 0 ,norm='l1', robust_gamma = 0.00015):

	#parameters
	epocs = sio.loadmat(epocs)['epocs'][0][0]
	#gamma_logscale = [1,2,3,4,5,6,7,8,9,10,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,-17,-18]
	gamma_logscale = [-3,-4]
	print np.size(gamma_logscale)
	print  epocs
	print epocs*np.size(gamma_logscale)

	clustering_accuracy = np.zeros(epocs*np.size(gamma_logscale))
	calculate_purity = np.zeros(epocs*np.size(gamma_logscale))
	calculate_nmi = np.zeros(epocs*np.size(gamma_logscale))

	cont = 0
	for gamma in gamma_logscale:
	    for epocs in xrange(epocs):
		   print epocs
		   labels_true,labels_pred,features= KernelKMeans.robust_kernel(fileData, normalized_axis = normalized_axis,norm=norm,gamma=2**gamma, robust_gamma = robust_gamma)
		   clustering_accuracy[cont] = Metrics.calculate_clusteringAccuracy(labels_true,labels_pred)
		   calculate_purity[cont],vector = Metrics.calculate_purity(labels_true,labels_pred)
		   calculate_nmi[cont] = Metrics.calculate_nmi(labels_true,labels_pred)
		   cont +=1

	sio.savemat('resultsRobust',{'clusteringAccuracy' : clustering_accuracy,'purityvec' : calculate_purity,'nmivec' : calculate_nmi})
