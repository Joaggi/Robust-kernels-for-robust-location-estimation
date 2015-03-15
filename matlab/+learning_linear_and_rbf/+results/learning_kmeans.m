userDir = char(java.lang.System.getProperty('user.home'));
userDir = '/home/jagallegom'
addpath(userDir);
addpath('/mnt/cuda/')


addpath('/home/jagallegom/Machine-Learning/algorithms/matlab/')
addpath('/home/jagallegom/Machine-Learning/dataset/')
addpath('/home/jagallegom/Machine-Learning/matlab/')

import learning_linear_and_rbf.experiments.learning_linear_and_rbf


options.vect = 2.^(0);
options.dataset.name= 'att'
options.dataset.dataset= 'data'
options.dataset.labels= 'labels'

% options.epocs = 100
% 
% options.algorithm.kernel = 'linear'
% options.algorithm.iter=3000;
% options.algorithm.dis=1;
% options.algorithm.residual=1e-70;
% options.algorithm.tof=1e-70;
% options.algorithm.initialization='random';
% options.algorithm.random = 1
% options.algorithm.name = 'kmeans'
% 
% options.preprocessing = 'normalize_by_range'
% 
% % learning_linear_and_rbf(options)
% 
% 
% options.vect = 2.^(0);
% options.dataset.name= 'balance_scale'
% options.dataset.dataset= 'data'
% options.dataset.labels= 'labels'
% 
% options.epocs = 100
% 
% options.algorithm.kernel = 'linear'
% options.algorithm.iter=3000;
% options.algorithm.dis=1;
% options.algorithm.residual=1e-70;
% options.algorithm.tof=1e-70;
% options.algorithm.initialization='random';
% options.algorithm.random = 1
% options.algorithm.name = 'kmeans'
% 
% options.preprocessing = 'normalize_by_range'
% 
% learning_linear_and_rbf(options)
% 
% 
% 
% options.vect = 2.^(0);
% options.dataset.name= 'glass'
% options.dataset.dataset= 'data'
% options.dataset.labels= 'labels'
% 
% options.epocs = 100
% 
% options.algorithm.kernel = 'linear'
% options.algorithm.iter=3000;
% options.algorithm.dis=1;
% options.algorithm.residual=1e-70;
% options.algorithm.tof=1e-70;
% options.algorithm.initialization='random';
% options.algorithm.random = 1
% options.algorithm.name = 'kmeans'
% 
% options.preprocessing = 'normalize_by_range'
% 
% learning_linear_and_rbf(options)
% 
% 
% 
% options.vect = 2.^(0);
% options.dataset.name= 'jaffe'
% options.dataset.dataset= 'data'
% options.dataset.labels= 'labels'
% 
% options.epocs = 100
% 
% options.algorithm.kernel = 'linear'
% options.algorithm.iter=3000;
% options.algorithm.dis=1;
% options.algorithm.residual=1e-70;
% options.algorithm.tof=1e-70;
% options.algorithm.initialization='random';
% options.algorithm.random = 1
% options.algorithm.name = 'kmeans'
% 
% options.preprocessing = 'normalize_by_range'
% 
% learning_linear_and_rbf(options)
% 
% 
% 
% options.vect = 2.^(0);
% options.dataset.name= 'movement_libras'
% options.dataset.dataset= 'data'
% options.dataset.labels= 'labels'
% 
% options.epocs = 100
% 
% options.algorithm.kernel = 'linear'
% options.algorithm.iter=3000;
% options.algorithm.dis=1;
% options.algorithm.residual=1e-70;
% options.algorithm.tof=1e-70;
% options.algorithm.initialization='random';
% options.algorithm.random = 1
% options.algorithm.name = 'kmeans'
% 
% options.preprocessing = 'normalize_by_range'
% 
% learning_linear_and_rbf(options)
% 
% 
% 
% options.vect = 2.^(0);
% options.dataset.name= 'wineq'
% options.dataset.dataset= 'data'
% options.dataset.labels= 'labels'
% 
% options.epocs = 100
% 
% options.algorithm.kernel = 'linear'
% options.algorithm.iter=3000;
% options.algorithm.dis=1;
% options.algorithm.residual=1e-70;
% options.algorithm.tof=1e-70;
% options.algorithm.initialization='random';
% options.algorithm.random = 1
% options.algorithm.name = 'kmeans'
% 
% options.preprocessing = 'normalize_by_range'
% 
% learning_linear_and_rbf(options)

% 
% 
% options.vect = 2.^(0);
% options.dataset.name= 'mnist_2k'
% options.dataset.dataset= 'data_train_2000'
% options.dataset.labels= 'labels_train_2000'
% 
% options.epocs = 100
%   
% options.algorithm.kernel = 'linear'
% options.algorithm.iter=3000;
% options.algorithm.dis=1;
% options.algorithm.residual=1e-70;
% options.algorithm.tof=1e-70;
% options.algorithm.initialization='random';
% options.algorithm.random = 1
% options.algorithm.name = 'kmeans'
% 
% options.preprocessing = 'normalize_by_range'
% 
% learning_linear_and_rbf(options)



options.vect = 2.^(0);
options.dataset.name= 'abalone'
options.dataset.dataset= 'data'
options.dataset.labels= 'labels'

options.epocs = 100

options.algorithm.kernel = 'linear'
options.algorithm.iter=3000;
options.algorithm.dis=1;
options.algorithm.residual=1e-70;
options.algorithm.tof=1e-70;
options.algorithm.initialization='random';
options.algorithm.random = 1
options.algorithm.name = 'kmeans'

options.preprocessing = 'normalize_by_range'

learning_linear_and_rbf(options)

options.vect = 2.^(0);
options.dataset.name= 'ar'
options.dataset.dataset= 'data'
options.dataset.labels= 'labels'

options.epocs = 100

options.algorithm.kernel = 'linear'
options.algorithm.iter=3000;
options.algorithm.dis=1;
options.algorithm.residual=1e-70;
options.algorithm.tof=1e-70;
options.algorithm.initialization='random';
options.algorithm.random = 1
options.algorithm.name = 'kmeans'

options.preprocessing = 'normalize_by_range'

learning_linear_and_rbf(options)
