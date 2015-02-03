import learning_linear_and_rbf.experiments.kmeans.learning_kmeans

options.vect = -20:25;
options.dataset.name= 'mnist'
options.dataset.dataset= 'images'
options.dataset.labels= 'labels_train_data'

options.epocs = 1

learning_kmeans(options)
