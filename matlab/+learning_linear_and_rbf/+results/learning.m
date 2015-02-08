import learning_linear_and_rbf.experiments.gaussian_kernel_cnmf.learning_gaussian_kernel_cnmf

options.vect = 2.^(0);
options.dataset.name= 'att'
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

learning_gaussian_kernel_cnmf(options)
