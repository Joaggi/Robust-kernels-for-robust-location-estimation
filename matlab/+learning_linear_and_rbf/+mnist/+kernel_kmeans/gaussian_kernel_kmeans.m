import learning_linear_and_rbf.experiments.gaussian_kernel_kmeans.learning_gaussian_kernel_kmeans

options.vect = 2.^(-25:25);
options.dataset.name= 'mnist'
options.dataset.dataset= 'images'
options.dataset.labels= 'labels_train_data'

options.epocs = 1

options.algorithm.kernel = 'rbf'
options.algorithm.iter=3000;
options.algorithm.dis=1;
options.algorithm.residual=1e-70;
options.algorithm.tof=1e-70;
options.algorithm.initialization='random';

learning_gaussian_kernel_kmeans(options)
