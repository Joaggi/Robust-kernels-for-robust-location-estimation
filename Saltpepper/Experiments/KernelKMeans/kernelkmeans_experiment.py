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

def kernelkmeans(fileData, epocs ,normalized_axis = 0 ,norm='l1'):

	#parameters
	epocs = sio.loadmat(epocs)['epocs'][0][0]

	gamma_logscale = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,-17,-18]
	print np.size(gamma_logscale)
	print  epocs
	print epocs*np.size(gamma_logscale)
	labels_total = []
    
    
	clustering_accuracy = np.zeros(epocs*np.size(gamma_logscale))
	calculate_purity = np.zeros(epocs*np.size(gamma_logscale))
	calculate_nmi = np.zeros(epocs*np.size(gamma_logscale))

	clustering_accuracy_mean = np.zeros(np.size(gamma_logscale))
	calculate_purity_mean = np.zeros(np.size(gamma_logscale))
	calculate_nmi_mean = np.zeros(np.size(gamma_logscale))
    
	clustering_accuracy_sd = np.zeros(np.size(gamma_logscale))
	calculate_purity_sd = np.zeros(np.size(gamma_logscale))
	calculate_nmi_sd = np.zeros(np.size(gamma_logscale))
	cont = 0
	contador = 0
	for j,gamma in zip(range(34),gamma_logscale):
	    aux=cont
	    for epocs in xrange(30):
    		print epocs
    		labels_true,labels_pred,features= KernelKMeans.get_kernelKMeans(fileData, normalized_axis = normalized_axis,norm=norm,gamma=2**gamma)
            	labels_total.append(labels_pred)
            	if np.size(np.unique(labels_pred)) == np.size(np.unique(labels_true)):
                	clustering_accuracy[cont] = Metrics.calculate_clusteringAccuracy(labels_true,labels_pred)
                	calculate_purity[cont],vector = Metrics.calculate_purity(labels_true,labels_pred)
                	calculate_nmi[cont] = Metrics.calculate_nmi(labels_true,labels_pred)
                	cont+=1
            	contador+=1
	    clustering_accuracy_mean[j] = np.mean(clustering_accuracy[aux:cont-1])
	    calculate_purity_mean[j] =  np.mean(calculate_purity[aux:cont-1])
	    calculate_nmi_mean[j] =  np.mean(calculate_nmi[aux:cont-1])
  
	    clustering_accuracy_sd[j] = np.std(clustering_accuracy[aux:cont-1])
	    calculate_purity_sd[j] =  np.std(calculate_purity[aux:cont-1])
	    calculate_nmi_sd[j] =  np.std(calculate_nmi[aux:cont-1])
        
    
	sio.savemat('G:/Dropbox/Universidad/Machine Learning/Robustes/AR/KernelKMeans/results',{'labels_total' : labels_total ,
                                           'clusteringAccuracyVec' : clustering_accuracy,
                                           'purityVec' : calculate_purity,
                                           'nmiVec' : calculate_nmi , 
                                           'clusteringAccuracyMeanVec' : clustering_accuracy_mean,
                                           'purityMeanVec' : calculate_purity_mean, 
                                           'nmiMeanVec' : calculate_nmi_mean, 
                                           'clusteringAccuracyStdVec' : clustering_accuracy_sd , 
                                           'purityStdVec' :calculate_purity_sd ,
                                           'nmiStdVec':calculate_nmi_sd})
