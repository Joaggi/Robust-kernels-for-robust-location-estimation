# -*- coding: utf-8 -*-
"""
Created on Wed May 14 17:05:59 2014

@author: Alejandro
"""
from GaussianExperiment import GaussianExperiment
#[Experiment 1]
def experiment(n_samples, n_outliers, n_clusters, n_features, n_experiment):
    gaussianExperiment = GaussianExperiment(n_samples = n_samples, \
        n_outliers = n_outliers, n_clusters = n_clusters, \
        n_features = n_features, n_experiment = n_experiment)

    X,y = gaussianExperiment.generate_data()
    X_contaminated = gaussianExperiment.generate_contamination()
    gaussianExperiment.show_graph_3d(X,y,X_contaminated,3)
    #gaussianExperiment.save_data()
    
experiment(n_samples= 300 , n_outliers = 150 , n_clusters = 4 \
    , n_features =3 , n_experiment = 1)
    
experiment(n_samples= 300 , n_outliers = 150 , n_clusters = 4 \
    , n_features =3 , n_experiment = 2)