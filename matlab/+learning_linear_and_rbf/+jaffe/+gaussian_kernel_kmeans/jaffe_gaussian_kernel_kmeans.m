import learning_linear_and_rbf.experiments.gaussian_kernel_kmeans.learning_gaussian_kernel_kmeans

options.vect = 2.^(-20:20);
options.dataset.name= 'jaffe'
options.dataset.dataset= 'data'
options.dataset.labels= 'labels'

options.epocs = 20

options.preprocessing='normalize_by_range'

options.algorithm.kernel = 'rbf'
options.algorithm.iter=3000;
options.algorithm.dis=1;
options.algorithm.residual=1e-70;
options.algorithm.tof=1e-70;
options.algorithm.initialization='random';

learning_gaussian_kernel_kmeans(options)
