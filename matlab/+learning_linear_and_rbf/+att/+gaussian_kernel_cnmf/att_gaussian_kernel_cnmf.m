import learning_linear_and_rbf.experiments.gaussian_kernel_cnmf.learning_gaussian_kernel_cnmf

options.vect = 2.^(13); 
options.dataset.name= 'ar'
options.dataset.dataset= 'data'
options.dataset.labels= 'labels'

options.epocs = 10

options.algorithm.kernel = 'linear'
options.algorithm.iter=3000;
options.algorithm.dis=1;
options.algorithm.residual=1e-70;
options.algorithm.tof=1e-70;
options.algorithm.initialization='kmeans';
options.algorithm.random = 1
options.algorithm.name = 'nmf'

options.preprocessing = 'raw'

learning_gaussian_kernel_cnmf(options)