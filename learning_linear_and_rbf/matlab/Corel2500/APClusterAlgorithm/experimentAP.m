% Affinity Propagation clustering
nrun = 2000;     % max iteration times, default 2000
nconv = 100;      % convergence condition, default 50
lam = 0.90;      % damping factor, default 0.9
splot = 'noplot';
%splot = 'plot'; % observing a clustering process when it is on

load('dat');
dat;
k = 20;
addpath 'G:\Dropbox\Universidad\Machine Learning\Algorithms\Matlab'
X = dat(:,1:k*100);
X= L1norm(X);
X = X';

M = similarity_euclid(X); 
M = -M


[labels,netsim,dpsim,expref,pref,lowp,highp]= apclusterK(M,k,0,nrun,nconv,lam,splot,-100,40,k-3,k+3)

vectorY = zeros(size(idx));
for i =1:k
vectorY(1+(i-1)*100:100*i)= i;
end

% NC = length(unique(labelid));
[C, idx] = ind2cluster(labels);


[purity,purityVector] = calculate_purity(idx,vectorY,k);