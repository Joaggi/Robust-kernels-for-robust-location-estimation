[value position] = max(purityMeanVec)

purityStdVec = std(purityVec(purityVec((position-1)*30+1:((position)+1)*30)~=0))

purityStdVec = std(purityVec(purityVec~=0))

purityStdVec = std(purityVec(purityVec((position-1)*6+1:(position)*6)~=0))



[value position] = max(nmiMeanVec)

nmiStdVec = std(nmiVec(nmiVec((position-1)*30+1:(position)*30)~=0))

nmiStdVec = std(nmiVec(nmiVec~=0))


[value position] = max(clusteringAccuracyMeanVec)

clusteringAccuracyStdVec = std(clusteringAccuracyVec(clusteringAccuracyVec((position-1)*30+1:(position)*30)~=0))

clusteringAccuracyStdVec = std(clusteringAccuracyVec(clusteringAccuracyVec~=0))

clusteringAccuracyStdVec = std(clusteringAccuracyVec(clusteringAccuracyVec((position-1)*6+1:position*6)~=0))



[purity position] = max(purityMeanVec)
purityStdVec = std(purityVec(purityVec~=0))
[nmi position] = max(nmiMeanVec)
nmiStdVec = std(nmiVec(nmiVec~=0))
[clustering position] = max(clusteringAccuracyMeanVec)
clusteringAccuracyStdVec = std(clusteringAccuracyVec(clusteringAccuracyVec~=0))


[purity position] = max(purityMeanVec)
purityStdVec = std(purityVec(purityVec((position-1)*6+1:(position)*6)~=0))
[nmi position] = max(nmiMeanVec)
nmiStdVec = std(nmiVec(nmiVec((position-1)*6+1:position*6)~=0))
[accuracy position] = max(clusteringAccuracyMeanVec)
clusteringAccuracyStdVec = std(clusteringAccuracyVec(clusteringAccuracyVec((position-1)*6+1:position*6)~=0))
