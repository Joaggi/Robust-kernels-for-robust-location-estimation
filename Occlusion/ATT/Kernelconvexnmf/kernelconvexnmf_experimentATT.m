addpath '../../../Algorithms/Matlab';
addpath '../../../Occlussion/Experiments/Kernelconvexnmf';
vect = [6,7,8,9,10,11,12,13]
kernelconvexnmf_experiment('../../../Occlussion/ATT','att_occlusion_contamination','att_occlusion_contamination',vect,strcat('Experimento ',date))
