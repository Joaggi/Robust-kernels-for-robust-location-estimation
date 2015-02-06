epocs = 30
purityVec = zeros(epocs,1);
clusteringAccuracyVec = zeros(epocs,1);
nmiVec = zeros(epocs,1);

for i=1:30
     [results,confus] = ClusteringMeasure(labels,labels_total(i,:))
     purityVec(i) = results(3) 
     clusteringAccuracyVec(i) = results(1)
     nmiVec(i) = results(2)
     
end

purityMeanVec = mean(purityVec)
clusteringAccuracyMeanVec = mean(clusteringAccuracyVec)
nmiMeanVec = mean(nmiVec)

purityStdVec = std(purityVec)
clusteringAccuracyStdVec = std(clusteringAccuracyVec)
nmiStdVec = std(nmiVec)