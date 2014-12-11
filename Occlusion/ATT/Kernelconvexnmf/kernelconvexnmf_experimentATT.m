addpath '../../../Algorithms/Matlab';
addpath '../../../Occlusion/Experiments/Kernelconvexnmf';
vect = [6,7,8,9,10,11,12,13]
kernelconvexnmf_experiment('../../../Occlusion/ATT','ATT_occlusion_contamination','ATT_occlusion_contamination',vect,strcat('Experimento ',date,' 2'))
