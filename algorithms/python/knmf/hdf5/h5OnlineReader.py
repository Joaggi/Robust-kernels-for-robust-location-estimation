# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 11:24:39 2013

@author: Jorge Vanegas
"""

import h5py
import numpy as np
import random

class HDF5OnlineReader:
    """Secuencial reader for HDF5 multimodal datasets             
    """
    minibatch_size = 0
    minibatch_index = -1
    minibatch_number = -1
    file_name = None
    dataset_paths = []
    h5_file = None
    datasets = []
    samples = -1
    indices = []
    shuffle = False
    last_minibatch_size = 0
    
    
    def __init__(self, file_name, minibatch_size, shuffle, *dataset_paths):
        """Construct a new HDF5OnlineReader instance.
        
        :param int minibatch: minibatch size
        :param str file_name: valid file name
        :param dataset_paths: strings with paths for each modality 
        """
        
        self.minibatch_size = minibatch_size
        self.file_name = file_name
        self.dataset_paths = dataset_paths
        self.shuffle = shuffle
        self.h5_file = h5py.File(file_name, 'r')
        
        for dp in dataset_paths:
            if self.samples == -1:
                self.samples = self.h5_file[dp].shape[0]
                self.minibatch_number = self.samples / self.minibatch_size
                self.last_minibatch_size = self.samples % self.minibatch_size
                if (self.last_minibatch_size != 0):
                    self.minibatch_number += 1
            elif self.h5_file[dp].shape[0] != self.samples:
                raise Exception('Datasets size doesn\'t match')
            self.datasets.append(self.h5_file[dp])    
            self.generate_indices()
        
        
    def generate_indices(self):
        """Generates a list of random indices"""
        
        self.indices = np.arange(self.samples)
        if self.shuffle:
            random.shuffle(self.indices)
        
    def get_minibatch(self, minibatch_index):
        """Returns the i-th multimodal minibatch
        
        :param int minibatch_index: index of required  multimodal minibatch
        :return: list of minibatches
        """
        
        start_index = minibatch_index * self.minibatch_size
        stop_index = start_index + self.minibatch_size
        modalities = []
        for dataset in self.datasets:
            modality = np.zeros((self.get_minibatc_size(stop_index), 
                                 dataset.shape[1],), dtype=float)
            i = 0;
            for index in self.indices[start_index:stop_index]:
                modality[i,:] = dataset[index,:]
                i += 1;
            modalities.append(modality)
        return modalities
            
    def get_next_minibatch(self):
        """Returns the next multimodal minibatch
        """
        
        self.minibatch_index += 1
        return self.get_minibatch(self.minibatch_index)
        
    def has_more_Data(self):
        """Return True if there are more minibatches to read
        """
        
        if self.minibatch_index + 1 < self.minibatch_number:
            return True
        return False
        
    def get_minibatc_size(self, stop_index):
        """return the next minibatch taking into account the number 
        of remaining examples
        """
        if stop_index <= self.samples:
            return self.minibatch_size
        return self.last_minibatch_size
        
            
        
        
        
        
        
        
        
        
    
    
