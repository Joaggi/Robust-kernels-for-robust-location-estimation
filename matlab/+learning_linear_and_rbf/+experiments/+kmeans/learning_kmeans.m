function correct = learning_kmeans(options)

c = clock;
nameresults = strcat(num2str(c(1)),num2str(c(2)),num2str(c(3)),'_',num2str(c(4)),'-',num2str(c(5)));

load(options.dataset.name,'-mat', options.dataset.dataset, options.dataset.labels)

data = eval(options.dataset.dataset);
labels = eval(options.dataset.labels);


options.k = length(unique(labels));
X = data;
% X= L1norm(data);
% X = X'
% X= normalizeByRange(data,1);

cont = 1


purityVec = zeros(options.epocs*length(options.vect),1);
clusteringAccuracyVec = zeros(options.epocs*length(options.vect),1);
nmiVec = zeros(options.epocs*length(options.vect),1);
labels_total = zeros(options.epocs*length(options.vect),length(labels));

purityMeanVec = zeros(length(options.vect),1);
clusteringAccuracyMeanVec = zeros(length(options.vect),1);
nmiMeanVec = zeros(length(options.vect),1);

puritySdVec = zeros(length(options.vect),1);
clusteringAccuracySdVec = zeros(length(options.vect),1);
nmiSdVec = zeros(length(options.vect),1);


l = 1
  
while(l<=length(options.vect))
    aux = cont
    options.algorithm.param = 2^options.vect(l);
    for i=1:options.epocs
            tic
            labels_pred = kmeans(X,options.k,'Emptyaction','singleton');;
            toc
            aux2 = 1;
            labels_total(cont,:) = labels_pred;
            [result,~] = ClusteringMeasure(labels,labels_pred)
            clusteringAccuracyVec(cont) = result(1);
            purityVec(cont) = result(3);
            nmiVec(cont) = result(2);
        cont = cont + 1
    end
    purityMeanVec(l) = mean(purityVec(aux:cont-1));
    clusteringAccuracyMeanVec(l) = mean(clusteringAccuracyVec(aux:cont-1));
    nmiMeanVec(l) = mean(nmiVec(aux:cont-1));
    puritySdVec(l) = std(purityVec(aux:cont-1));
    clusteringAccuracySdVec(l) = std(clusteringAccuracyVec(aux:cont-1));
    nmiSdVec(l) = std(nmiVec(aux:cont-1));
    l=l+1
    save(nameresults, 'clusteringAccuracyVec','purityVec','nmiVec','purityMeanVec','clusteringAccuracyMeanVec','nmiMeanVec','labels_total','l','puritySdVec','clusteringAccuracySdVec','nmiSdVec')
    
end
save(nameresults, 'clusteringAccuracyVec','purityVec','nmiVec','purityMeanVec','clusteringAccuracyMeanVec','nmiMeanVec','labels_total','l','puritySdVec','clusteringAccuracySdVec','nmiSdVec')
end
