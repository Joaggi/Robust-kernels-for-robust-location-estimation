function correct = kmeans_experiment(filedata,dataset,datalabels,nameresults)

if(nargin < 4)
    nameresults = 'results'
end


addpath(filedata);
load(dataset)
load(datalabels)
load('parameters')

addpath 'G:/Dropbox/Universidad/Machine Learning/Algorithms/Matlab';

% addpath '/home/jagallegom/Algorithms/Matlab';
% addpath '/home/jagallegom/Algorithms/Matlab/nmfv1_4';

k = length(unique(labels));
% X= normalizeByRange(data);



purityMeanVec = zeros(10,1)
clusteringAccuracyMeanVec = zeros(10,1)
nmiMeanVec = zeros(10,1)
puritySdVec = zeros(10,1)
clusteringAccuracySdVec = zeros(10,1)
nmiSdVec = zeros(10,1)


for j=1:10
    cont = 1
    X = data(:,:,j)
    purityVec = zeros(epocs,1);
    clusteringAccuracyVec = zeros(epocs,1);
    nmiVec = zeros(epocs,1);
    labels_total = zeros(epocs,length(labels));
    for i=1:epocs
%         try        
            k = length(unique(labels))
            labels_pred=kmeans(X,k,'Emptyaction','singleton'); % k-mean clustering, get idx=[1,1,2,2,1,3,3,1,2,3]
	        labels_total(cont,:) = labels_pred;
            [result,confus] = ClusteringMeasure(labels,labels_pred)
            clusteringAccuracyVec(cont) = result(1)
            purityVec(cont) = result(3)
            nmiVec(cont) = result(2)
%         catch
%             sprintf('Ha ocurrido un error')
%             purityVec(cont) = 0;
%             nmiVec(cont) = 0;
%             clusteringAccuracyVec(cont) = 0;
%         end
        cont = cont + 1
    end

    purityMeanVec(j) = mean(purityVec)
    clusteringAccuracyMeanVec(j) = mean(clusteringAccuracyVec)
    nmiMeanVec(j) = mean(nmiVec)
    puritySdVec(j)= std(purityVec)
    clusteringAccuracySdVec(j) = std(clusteringAccuracyVec)
    nmiSdVec(j) = std(nmiVec)


    
    
save(nameresults, 'clusteringAccuracyVec','purityVec','nmiVec','purityMeanVec','clusteringAccuracyMeanVec','nmiMeanVec','labels_total','puritySdVec','clusteringAccuracySdVec','nmiSdVec')
end
