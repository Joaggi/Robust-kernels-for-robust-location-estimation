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
X= normalizeByRange(data);
% X = data

purityVec = zeros(epocs,1);
clusteringAccuracyVec = zeros(epocs,1);
nmiVec = zeros(epocs,1);
labels_total = zeros(epocs,length(labels));


cont = 1

    for i=1:epocs
        try        
            k = length(unique(labels))
%             K = linearKernel(X,X);
%             [labels_pred, energy] = knkmeans(K, k+50);
%             contRecur = 0
%             while length(unique(labels_pred)) ~= k
%                 [labels_pred, energy] = knkmeans(K, k+50);
%                 if contRecur >30
%                     break
%                 end
%                 contRecur  = contRecur +1
%             end
            labels_pred=kmeans(X,k,'Emptyaction','singleton'); % k-mean clustering, get idx=[1,1,2,2,1,3,3,1,2,3]
	        labels_total(cont,:) = labels_pred;
            [result,confus] = ClusteringMeasure(labels,labels_pred)
            clusteringAccuracyVec(cont) = result(1)
            purityVec(cont) = result(3)
            nmiVec(cont) = result(2)
        catch
            sprintf('Ha ocurrido un error')
            purityVec(cont) = 0;
            nmiVec(cont) = 0;
            clusteringAccuracyVec(cont) = 0;
        end
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
