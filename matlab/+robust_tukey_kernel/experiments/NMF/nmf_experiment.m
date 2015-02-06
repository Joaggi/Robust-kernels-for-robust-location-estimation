function correct = nmf_experiment(filedata,dataset,datalabels,nameresults)

if(nargin < 4)
    nameresults = 'results'
end


addpath(filedata);
load(dataset)
load(datalabels)
load('parameters')

labels = typecast(labels, 'double')

addpath 'G:/Dropbox/Universidad/Machine Learning/Algorithms/Matlab';
addpath 'G:/Dropbox/Universidad/Machine Learning/Algorithms/Matlab/nmfv1_4';

% addpath '/home/jagallegom/Algorithms/Matlab';
% addpath '/home/jagallegom/Algorithms/Matlab/nmfv1_4';

k = length(unique(labels));
%X= L1norm(data);
X= data;

option.reorder=true;
option.algorithm='nmfrule';
option.iterations=3000;
option.iter=3000;
option.distance = 'ls';

purityVec = zeros(epocs,1);
clusteringAccuracyVec = zeros(epocs,1);
nmiVec = zeros(epocs,1);
labels_total = zeros(epocs*21,length(labels));

cont = 1

    for i=1:epocs
%         try
            %addpath '/home/jagallegom/Algorithms/Matlab/nmfv1_4';
            addpath 'G:/Dropbox/Universidad/Machine Learning/Algorithms/Matlab/nmfv1_4';
        
            k = length(unique(labels))
            labels_pred = NMFCluster(X',k,option);
	        labels_total(cont,:) = labels_pred;
            labels_pred
            %addpath '/home/jagallegom/Algorithms/Matlab';
            addpath 'G:/Dropbox/Universidad/Machine Learning/Algorithms/Matlab';
            
            [result,confus] = ClusteringMeasure(labels,labels_pred)
            clusteringAccuracyVec(cont) = result(1);
            nmiVec(cont) = result(2);
            purityVec(cont) = result(3);
%         catch
%             sprintf('Ha ocurrido un error')
%             purityVec(cont) = 0;
%             nmiVec(cont) = 0;
%             clusteringAccuracyVec(cont) = 0;
% 
%         end
        cont = cont + 1
    end

purityMeanVec = mean(purityVec)
clusteringAccuracyMeanVec = mean(clusteringAccuracyVec)
nmiMeanVec = mean(nmiVec)
puritySdVec= std(purityVec)
clusteringAccuracySdVec = std(clusteringAccuracyVec)
nmiSdVec = std(nmiVec)
    
    
save(nameresults, 'clusteringAccuracyVec','purityVec','nmiVec','purityMeanVec','clusteringAccuracyMeanVec','nmiMeanVec','labels_total','puritySdVec','clusteringAccuracySdVec','nmiSdVec')
end
