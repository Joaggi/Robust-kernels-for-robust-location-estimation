function correct = kernelconvexnmf_experiment_clusteringmeasures(filedata,dataset,datalabels,fileresults)

addpath(filedata);
addpath(fileresults);
load('results.mat')
load(datalabels)
addpath 'C:/Users/Joag/Dropbox/Universidad/Machine Learning/Algorithms/Matlab';

epocs = 30
vect = [1]
l = 1

purityVec = zeros(epocs*length(vect),1);
clusteringAccuracyVec = zeros(epocs*length(vect),1);
nmiVec = zeros(epocs*length(vect),1);
labels_total = zeros(epocs*length(vect),length(labels));

purityMeanVec = zeros(length(vect),1);
clusteringAccuracyMeanVec = zeros(length(vect),1);
nmiMeanVec = zeros(length(vect),1);


    for i=1:epocs

            labels_pred = labels_total(i,:)
            purityVec(cont) = purity(labels,labels_pred)
            clusteringAccuracyVec(cont) = clusteringaccuracy(labels,labels_pred)
            nmiVec(cont) = nmi(labels,labels_pred)
    end
    purityMeanVec(l) = mean(purityVec(aux:cont-1))
    clusteringAccuracyMeanVec(l) = mean(clusteringAccuracyVec(aux:cont-1))
    nmiMeanVec(l) = mean(nmiVec(aux:cont-1))
    
save results clusteringAccuracyVec purityVec nmiVec purityMeanVec clusteringAccuracyMeanVec nmiMeanVec labels_total
end
