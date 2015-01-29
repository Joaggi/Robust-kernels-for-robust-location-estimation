

addpath '../../../Algorithms/Matlab';
addpath '../../../UniformContamination/Experiments/KMeans';

kmeans_experiment('../../../UniformContamination/Abalone','abalone','abalone_labels',strcat('Experimento ',date))
