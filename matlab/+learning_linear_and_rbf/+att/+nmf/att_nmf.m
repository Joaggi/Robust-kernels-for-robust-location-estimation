import learning_linear_and_rbf.experiments.nmf.learning_nmf

options.vect = 2.^(-20:20);
options.dataset.name= 'att'
options.dataset.dataset= 'data'
options.dataset.labels= 'labels'

options.epocs = 20

options.preprocessing='normalize_by_range'

options.algorithm.reorder=true;
options.algorithm.algorithm='nmfrule';
options.algorithm.iterations=3000;
options.algorithm.iter=3000;
options.algorithm.distance = 'ls';
options.algorithm.initialization = 'kmeans';

learning_nmf(options)