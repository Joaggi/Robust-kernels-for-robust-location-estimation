function correct = ap_experiment(filedata,dataset,datalabels)

% Affinity Propagation clustering
nrun = 2000;     % max iteration times, default 2000
nconv = 100;      % convergence condition, default 50
lam = 0.80;      % damping factor, default 0.9
splot = 'noplot';
%splot = 'plot'; % observing a clustering process when it is on


addpath(filedata);
load(dataset);
load(datalabels);
load('parameters')

addpath 'G:/Dropbox/Universidad/Machine Learning/Algorithms/Matlab';

% addpath '/home/jagallegom/Algorithms/Matlab';

k = max(size(unique(labels)));
X= L1norm(data);


purityVec = zeros(epocs*21,1);
clusteringAccuracyVec = zeros(epocs*21,1);
nmiVec = zeros(epocs*21,1);
labels_total = zeros(epocs*21,length(labels));

purityMeanVec = zeros(21,1);
clusteringAccuracyMeanVec = zeros(21,1);
nmiMeanVec = zeros(21,1);

cont = 1
vect = [-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,0,1,2,3,4,5,6,7,8,9,10];
%vect = [-3,-4]


for l = 1:1
    aux = cont
    for i=1:epocs

            M = similarity_euclid(X); 
            M= -M
            k = length(unique(labels))
            [la,netsim,dpsim,expref,pref,lowp,highp] = apclusterK(M,k,0,nrun,nconv,lam,splot,-30,30,k-1,k+1)
            [C, labels_pred] = ind2cluster(la)
            la
            C
	        labels_total(cont,:) = labels_pred;
            clear energy''
            purityVec(cont) = purity(labels,labels_pred);
            clusteringAccuracyVec(cont) = clusteringaccuracy(labels,labels_pred);
            nmiVec(cont) = nmi(labels,labels_pred);

        cont = cont + 1
    end
    purityMeanVec(l) = mean(purityVec(aux:cont-1))
    clusteringAccuracyMeanVec(l) = mean(clusteringAccuracyVec(aux:cont-1))
    nmiMeanVec(l) = mean(nmiVec(aux:cont-1))
    
end
save results clusteringAccuracyVec purityVec nmiVec purityMeanVec clusteringAccuracyMeanVec nmiMeanVec labels_total
end
