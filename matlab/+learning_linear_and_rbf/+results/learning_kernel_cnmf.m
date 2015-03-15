userDir = char(java.lang.System.getProperty('user.home'));
userDir = '/home/jagallegom'
addpath(userDir);
addpath('/mnt/cuda/')


addpath(strcat(userDir,'/Machine-Learning/algorithms/matlab/'))
addpath(strcat(userDir,'/Machine-Learning/dataset/'))
addpath(strcat(userDir,'/Machine-Learning/matlab/'))

import learning_linear_and_rbf.experiments.learning_linear_and_rbf


options.vect = 2.^(-10:10);
options.dataset.name= 'att'
options.dataset.dataset= 'data'
options.dataset.labels= 'labels'

options.epocs = 1

options.algorithm.kernel = 'rbf'
options.algorithm.iter=150;
options.algorithm.dis=1;
options.algorithm.residual=1e-70;
options.algorithm.tof=1e-70;
options.algorithm.initialization='random';
options.algorithm.random = 1
options.algorithm.name = 'kernel_convex_nmf'

options.preprocessing = 'normalize_by_range'

learning_linear_and_rbf(options)


addpath('/home/jagallegom/Machine-Learning/algorithms/matlab/')
addpath('/home/jagallegom/Machine-Learning/dataset/')
addpath('/home/jagallegom/Machine-Learning/matlab/')

options.vect = 2.^(-2:2);
options.dataset.name= 'balance_scale'
options.dataset.dataset= 'data'
options.dataset.labels= 'labels'

options.epocs = 20

options.algorithm.kernel = 'rbf'
options.algorithm.iter=3000;
options.algorithm.dis=1;
options.algorithm.residual=1e-70;
options.algorithm.tof=1e-70;
options.algorithm.initialization='random';
options.algorithm.random = 1
options.algorithm.name = 'kernel_convex_nmf'

options.preprocessing = 'normalize_by_range'

learning_linear_and_rbf(options)

addpath('/home/jagallegom/Machine-Learning/algorithms/matlab/')
addpath('/home/jagallegom/Machine-Learning/dataset/')
addpath('/home/jagallegom/Machine-Learning/matlab/')


options.vect = 2.^(-2:2);
options.dataset.name= 'glass'
options.dataset.dataset= 'data'
options.dataset.labels= 'labels'

options.epocs = 20

options.algorithm.kernel = 'rbf'
options.algorithm.iter=3000;
options.algorithm.dis=1;
options.algorithm.residual=1e-70;
options.algorithm.tof=1e-70;
options.algorithm.initialization='random';
options.algorithm.random = 1
options.algorithm.name = 'kernel_convex_nmf'

options.preprocessing = 'normalize_by_range'

learning_linear_and_rbf(options)

addpath('/home/jagallegom/Machine-Learning/algorithms/matlab/')
addpath('/home/jagallegom/Machine-Learning/dataset/')
addpath('/home/jagallegom/Machine-Learning/matlab/')


options.vect = 2.^(-2:2);
options.dataset.name= 'iris'
options.dataset.dataset= 'data'
options.dataset.labels= 'labels'

options.epocs = 20

options.algorithm.kernel = 'rbf'
options.algorithm.iter=3000;
options.algorithm.dis=1;
options.algorithm.residual=1e-70;
options.algorithm.tof=1e-70;
options.algorithm.initialization='random';
options.algorithm.random = 1
options.algorithm.name = 'kernel_convex_nmf'

options.preprocessing = 'normalize_by_range'

learning_linear_and_rbf(options)

addpath('/home/jagallegom/Machine-Learning/algorithms/matlab/')
addpath('/home/jagallegom/Machine-Learning/dataset/')
addpath('/home/jagallegom/Machine-Learning/matlab/')


options.vect = 2.^(-2:2);
options.dataset.name= 'jaffe'
options.dataset.dataset= 'data'
options.dataset.labels= 'labels'

options.epocs = 20

options.algorithm.kernel = 'rbf'
options.algorithm.iter=3000;
options.algorithm.dis=1;
options.algorithm.residual=1e-70;
options.algorithm.tof=1e-70;
options.algorithm.initialization='random';
options.algorithm.random = 1
options.algorithm.name = 'kernel_convex_nmf'

options.preprocessing = 'normalize_by_range'

learning_linear_and_rbf(options)

addpath('/home/jagallegom/Machine-Learning/algorithms/matlab/')
addpath('/home/jagallegom/Machine-Learning/dataset/')
addpath('/home/jagallegom/Machine-Learning/matlab/')



options.vect = 2.^(-2:2);
options.dataset.name= 'movement_libras'
options.dataset.dataset= 'data'
options.dataset.labels= 'labels'

options.epocs = 20

options.algorithm.kernel = 'rbf'
options.algorithm.iter=3000;
options.algorithm.dis=1;
options.algorithm.residual=1e-70;
options.algorithm.tof=1e-70;
options.algorithm.initialization='random';
options.algorithm.random = 1
options.algorithm.name = 'kernel_convex_nmf'

options.preprocessing = 'normalize_by_range'

learning_linear_and_rbf(options)

addpath('/home/jagallegom/Machine-Learning/algorithms/matlab/')
addpath('/home/jagallegom/Machine-Learning/dataset/')
addpath('/home/jagallegom/Machine-Learning/matlab/')



options.vect = 2.^(-2:2);
options.dataset.name= 'wineq'
options.dataset.dataset= 'data'
options.dataset.labels= 'labels'

options.epocs = 20

options.algorithm.kernel = 'rbf'
options.algorithm.iter=3000;
options.algorithm.dis=1;
options.algorithm.residual=1e-70;
options.algorithm.tof=1e-70;
options.algorithm.initialization='random';
options.algorithm.random = 1
options.algorithm.name = 'kernel_convex_nmf'

options.preprocessing = 'normalize_by_range'

learning_linear_and_rbf(options)

addpath('/home/jagallegom/Machine-Learning/algorithms/matlab/')
addpath('/home/jagallegom/Machine-Learning/dataset/')
addpath('/home/jagallegom/Machine-Learning/matlab/')

options.vect = 2.^(-2:2);
options.dataset.name= 'mnist_2k'
options.dataset.dataset= 'train_data_2000'
options.dataset.labels= 'labels_train_data_2000'

options.epocs = 20

options.algorithm.kernel = 'rbf'
options.algorithm.iter=3000;
options.algorithm.dis=1;
options.algorithm.residual=1e-70;
options.algorithm.tof=1e-70;
options.algorithm.initialization='random';
options.algorithm.random = 1
options.algorithm.name = 'kernel_convex_nmf'

options.preprocessing = 'normalize_by_range'

learning_linear_and_rbf(options)


addpath('/home/jagallegom/Machine-Learning/algorithms/matlab/')
addpath('/home/jagallegom/Machine-Learning/dataset/')
addpath('/home/jagallegom/Machine-Learning/matlab/')



options.vect = 2.^(-2:2);
options.dataset.name= 'abalone'
options.dataset.dataset= 'data'
options.dataset.labels= 'labels'

options.epocs = 20

options.algorithm.kernel = 'rbf'
options.algorithm.iter=3000;
options.algorithm.dis=1;
options.algorithm.residual=1e-70;
options.algorithm.tof=1e-70;
options.algorithm.initialization='random';
options.algorithm.random = 1
options.algorithm.name = 'kernel_convex_nmf'

options.preprocessing = 'normalize_by_range'

learning_linear_and_rbf(options)
