current_path = '../../../../';


addpath(strcat(current_path, 'algorithms/matlab'));
addpath(strcat(current_path, 'learning_linear_and_rbf/matlab/experiments/Kernelconvexnmf'));

kernelconvexnmf_experiment(strcat(current_path, 'learning_linear_and_rbf/matlab/att'),'att','att',[1],date)
