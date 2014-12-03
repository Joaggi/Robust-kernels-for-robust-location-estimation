addpath '../../../Algorithms/Matlab';

addpath '../../../Experiments/KernelKMeans';

kernelkmeans_experiment('../../../UniformContamination/Abalone','abalone','abalone_labels',strcat('Experimento ',date))
