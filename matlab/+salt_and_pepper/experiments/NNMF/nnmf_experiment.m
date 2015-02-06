function correct = nnmf_experiment(filedata,dataset,datalabels)

addpath(filedata);
load(dataset)
load(datalabels)
load('parameters')

addpath 'G:/Dropbox/Universidad/Machine Learning/Algorithms/Matlab';

% addpath '/home/jagallegom/Algorithms/Matlab';

k = length(unique(labels));
X= L1norm(data);

purityVec = zeros(epocs,1);
clusteringAccuracyVec = zeros(epocs,1);
nmiVec = zeros(epocs,1);
labels_total = zeros(epocs*21,length(labels));

purityMeanVec =0;
clusteringAccuracyMeanVec = 0;
nmiMeanVec =0;

cont = 1

    for i=1:epocs
        try     
            k = length(unique(labels))
            [labels_pred,Xout,Aout,Yout] = NNMFCluster(X',k)
            labels_pred
            
	        labels_total(cont,:) = labels_pred;
            labels_pred
            purityVec(cont) = purity(labels,labels_pred)
            clusteringAccuracyVec(cont) = clusteringaccuracy(labels,labels_pred)
            nmiVec(cont) = nmi(labels,labels_pred)
        catch
            purityVec(cont) = 0;
            nmiVec(cont) = 0;
            clusteringAccuracyVec(cont) = 0;

        end
        cont = cont + 1
    end

purityMeanVec = mean(purityVec);
clusteringAccuracyMeanVec = mean(clusteringAccuracyVec);
nmiMeanVec = mean(nmiVec)
    
    
save results clusteringAccuracyVec purityVec nmiVec clusteringAccuracyMeanVec purityMeanVec nmiMeanVec labels_total
end
