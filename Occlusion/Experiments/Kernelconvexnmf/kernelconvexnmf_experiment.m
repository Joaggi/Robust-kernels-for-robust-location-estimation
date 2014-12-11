function kernelconvexnmf_experiment(filedata,dataset,datalabels,vect,nameresults)

addpath '../../../Algorithms/Matlab';
addpath '../../../Algorithms/Matlab/nmfv1_4';
addpath '../../../Datasets';
addpath(filedata);
load(dataset)
load(datalabels)

% 
epocs=6;


k = length(unique(labels));
X= data_real;
% [X,minimo,maximo]= normalizeByRange(data,1);
%  X= L1norm(data);

porcentajeContaminacion= 30;
[numContamination,clusteringAccuracyVec,clusteringAccuracyMeanVec,clusteringAccuracySdVec] =   definicionVariables(X,vect,epocs,porcentajeContaminacion)

size(clusteringAccuracyVec)

option.kernel = 'rbf';
option.iter=500;
option.dis=1;
option.residual=1e-4;
option.tof=1e-4;
option.initialization = 'random';


try
    load(nameresults);
catch
    j=1;
end

cont = 1;
for l = 1:length(vect)
    aux = cont;
    option.param = 2^vect(l);
    for i=1:epocs       
            [labelsPred,~,~,~,~,~,~] = kernelCNMFAlgorithm(X,k,option);
            [result,~] = ClusteringMeasure(labels,labelsPred);
            clusteringAccuracyVec(cont) = result(1)
            cont = cont + 1
    end
    clusteringAccuracyMeanVec(l) = mean(clusteringAccuracyVec(aux:cont-1));
    clusteringAccuracySdVec(l) = std(clusteringAccuracyVec(aux:cont-1));
end


[~, posMin] = max(clusteringAccuracyMeanVec);
s = vect(posMin);

option.param = 2^vect(posMin);

cont = 1;
while(j<=porcentajeContaminacion)
        aux = cont;
        for i=1:epocs      
            XwithContamination = data_real;
            randompermutation = randperm(size(data_contamination,1),floor(size(data_contamination,1)*(j/100)));
            XwithContamination(randompermutation,:) = data_contamination(randompermutation,:);
            k = length(unique(labels));
            [labelsPred,~,~,~,~,~,~] = kernelCNMFAlgorithm(XwithContamination,k,option);
            [result,~] = ClusteringMeasure(labels,labelsPred);
            clusteringAccuracyVec(cont) = result(1);
            cont = cont + 1
        end
        clusteringAccuracyMeanVec(l) = mean(clusteringAccuracyVec(aux:cont-1));
        clusteringAccuracySdVec(l) = std(clusteringAccuracyVec(aux:cont-1));        
        j=j+1
        save(nameresults,'clusteringAccuracyVec','clusteringAccuracyMeanVec','clusteringAccuracySdVec','clusteringAccuracyMeanVec','clusteringAccuracySdVec','X','j','s','option')
end


save(nameresults,'clusteringAccuracyVec','clusteringAccuracyMeanVec','clusteringAccuracySdVec','clusteringAccuracyMeanVec','clusteringAccuracySdVec','X','j','s','option')
end


function [numContamination,clusteringAccuracyVec,clusteringAccuracyMeanVec,clusteringAccuracySdVec] =   definicionVariables(X,vect,epocs,porcentajeContaminacion)
    numContamination = floor(porcentajeContaminacion/100*size(X,1));
    clusteringAccuracyVec = zeros(epocs*length(vect),1);
    clusteringAccuracyMeanVec = zeros(length(vect),1);
    clusteringAccuracySdVec = zeros(length(vect),1);
end

function [labelsPred,Xout,Aout,Yout,numIter,tElapsed,finalResidual] = kernelCNMFAlgorithm(X,k,option)
            [labelsPred,Xout,Aout,Yout,numIter,tElapsed,finalResidual] = kernelconvexnmfCluster(X',k,option);
            contRecur = 0;
            while length(unique(labelsPred)) ~= k
                [labelsPred,Xout,Aout,Yout,numIter,tElapsed,finalResidual] = kernelconvexnmfCluster(X',k,option);
                clear energy;
                if contRecur >15
                    break
                end
                contRecur  = contRecur +1;
            end
end


