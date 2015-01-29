

addpath '../../../Algorithms/Matlab';
addpath '../../../UniformContamination/Experiments/Kernelconvexnmf';

kernelconvexnmf_experiment('../../../UniformContamination/Abalone','abalone_contamination','abalone_contamination',strcat('Experimento ',date))
