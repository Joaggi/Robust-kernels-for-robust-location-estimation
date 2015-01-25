# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:59:10 2013

@author: Jorge Vanegas
"""

import h5py
import numpy as np
import sys

def plain2h5(dataset_name, dataset_paths):
       
    h5_file = h5py.File(dataset_name, 'w')
    
    for dataset_path_h5 in dataset_paths:
        print 'Exporting datset from plain text file \'', dataset_paths[dataset_path_h5], '\' to \'', dataset_path_h5, '\' in ', dataset_name, '...'
        
        matrix = read_plain_file(dataset_paths[dataset_path_h5])
        ds = h5_file.create_dataset(dataset_path_h5, matrix.shape, maxshape=(None,matrix.shape[1]))
        ds[...] = matrix
    
    h5_file.close()
     
def read_plain_file(file_path):
    matrix = []
    for line in open(file_path).readlines():
        data = map(float,line.split())
        matrix.append(data)
    return np.asmatrix(matrix)
    
if len(sys.argv) < 4:
    print 'Use: plain2h5.py dataset_name modality_name_1 modality_path_1 modality_name_2 modality_path_2 ... modality_name_N modality_path_N'
    sys.exit()

dataset_name = sys.argv[1]
dataset_paths = {}
i = 2
while i < len(sys.argv):
    dataset_paths[sys.argv[i]] = sys.argv[i + 1]
    i += 2

plain2h5(dataset_name, dataset_paths)   
    