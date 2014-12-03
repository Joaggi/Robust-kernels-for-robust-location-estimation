function correct = kernelconvexnmf_experiment(filedata,dataset,datalabels,nameresults)

addpath '../../../Algorithms/Matlab';
addpath '../../../Algorithms/Matlab/nmfv1_4';
addpath '../../../Datasets';
addpath(filedata);
load(dataset)
load(datalabels)
load('parameters')
% 
epocs=50


k = length(unique(labels));
% X= data;
X= normalizeByRange(data,1);
% X= L1norm(data);

cont = 1
vect = [-1,-2,-3,-4,-5,-6,0,1,2,3,4,5,6];
%vect = [-9]

porcentajeContaminacion= 30
[numContamination,vectorCentroids,realCentroids,centroids,vectorCentroidsContamination,bias,biasContamination,biasGlobal] =   definicionVariables(X,vect,labels,k,epocs,porcentajeContaminacion);
cont = 1


option.kernel = 'rbf';
option.iter=100;
option.dis=1;
option.residual=1e-4;
option.tof=1e-4;
option.initialization = 'kkmeans';


try
    load(nameresults);
    name = 1;
catch
    name = 0;
    j=1;
end

biasAux = inf

for l = 1:length(vect)
    aux = cont
    option.param = 2^vect(l);
    for i=1:epocs       
            [labelsPred,~,~,Yout,~,~,~] = kernelCNMFAlgorithm(X,k,option);
            vectorCentroids(:,:,cont) = (preW(X',Yout', 2^vect(l),20))';
            vectorCentroids(:,:,cont) = geneticAlgorithm(realCentroids,vectorCentroids(:,:,cont));
            cont = cont + 1
    end
    centroids(:,:,l) = mean(vectorCentroids(:,:,(l-1)*epocs+1:(l)*epocs),3);
    bias(l) = sum(sum(abs(centroids(:,:,l) - realCentroids)));
    
    if bias(l) < biasAux
        labelsReal = labelsPred;
        biasAux = bias(l);
    end
end


[value, posMin] = min(bias);
s = vect(posMin)
centroid = centroids(:,:,posMin)
centroidInitial = centroid;
biasInitial = sum(sum(abs(centroid - realCentroids)));

minFeature = min(X)-min(X)*0.5;
maxFeature = max(X)+max(X)*0.5;

%vectorContamination = unifrnd(repmat(minFeature,numContamination,1),repmat(maxFeature,numContamination,1),numContamination,length(minFeature));

option.param = 2^vect(posMin);

while(j<=porcentajeContaminacion)
        booleanVectorCentroids = ones(1,epocs);
        vectorCentroidsContamination = zeros(k,size(X,2),epocs);
        XwithContamination = [X;vectorContamination(1:floor(size(X,1)*(j/100)),:)];
        for i=1:epocs      
            try
                k = length(unique(labels));
                [~,~,~,Yout,~,~,~] = kernelCNMFAlgorithm(XwithContamination,k,option);
                vectorCentroidsContamination(:,:,i) = (preW(XwithContamination', Yout', 2^s,20))';
                vectorCentroidsContamination(:,:,i) = geneticAlgorithm(centroidInitial,vectorCentroidsContamination(:,:,i));
            catch
                sprintf('Ha ocurrido un error.');
                booleanVectorCentroids(i) = 0;
            end
        end
        
        centroidContamination = mean(vectorCentroidsContamination(:,:,booleanVectorCentroids==1),3);
        biasContamination(j) =  sum(sum(abs(centroidContamination - centroidInitial)));
        biasGlobal(j) = sum(sum(abs(centroidContamination - realCentroids)));
        j=j+1
        save(nameresults,'vectorCentroids','biasInitial','bias','realCentroids','X','vectorContamination','biasContamination','vectorCentroidsContamination','j','centroidInitial','s','labelsReal','option')
end


save(nameresults,'vectorCentroids','biasInitial','bias','realCentroids','X','vectorContamination','biasContamination','biasGlobal','vectorCentroidsContamination','j','centroidInitial','s','labelsReal','option')
end


function centroids = get_centroids_real(labels,data)
    unique_labels = unique(labels);
    k = length(unique(labels))
    centroids = zeros(k,size(data,2));
    for i=1:k
        centroids(i,:)= mean(data(labels == unique_labels(i),:));
    end
end

function centroids = get_centroids(labels,data)
    unique_labels = unique(labels);
    k = length(unique(labels));
    centroids = zeros(k,size(data,2));
    for i=1:k
        centroids(i,:)= mean(data(labels == unique_labels(i),:));
    end
end


function [numContamination,vectorCentroids,realCentroids,centroids,vectorCentroidsContamination,bias,biasContamination,biasGlobal] =   definicionVariables(X,vect,labels,k,epocs,porcentajeContaminacion)
    numContamination = floor(porcentajeContaminacion/100*size(X,1));
    vectorCentroids = zeros(k,size(X,2),epocs*length(vect));
    realCentroids = get_centroids(labels,X);
    centroids  =  zeros(k,size(X,2),length(vect));
    vectorCentroidsContamination = zeros(k,size(X,2),epocs*length(vect));
    bias = zeros(length(vect),1);
    biasContamination = zeros(porcentajeContaminacion,1);
    biasGlobal = zeros(porcentajeContaminacion,1);

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


