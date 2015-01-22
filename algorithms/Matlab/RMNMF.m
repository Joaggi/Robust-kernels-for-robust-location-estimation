function [idx,H]=RMNMF(X,Y,K,mu,lambda,pho)
% % this function implements the method described in the TKDD journal paper 
%  "Robust Manifold Non-Negative Matrix Factorization" by Jin Huang,
%   Feiping Nie, Heng Huang and Chris Ding. 
%  Please contact Jin Huang (huangjinsuzhou@gmail.com) if you had any question or 
%  found any bug in the code. Thanks!

% Input:
% X is row-based data matrix of size n by p
% Y is a vector of label of size n by 1
% K is the number of clusters initialized
% mu is the initial parameter value via ALM
% lambda is the regularity parameter

% Output:
% result:ACC, NMI, Purity 



% In this code, we use affinity graph construction code from others, please go
% to http://www.cad.zju.edu.cn/home/dengcai/Data/DimensionReduction.html
% to download constructW function and understand the parameter setting here
% you can also construct the affinity graph in your own way
if nargin == 5
pho=1.05;
end
lambda;

p=size(X,1);
n=size(X,2);

% This part of code constructs the Laplacian Graph
options = [];
options.Metric = 'Euclidean';
options.NeighborMode = 'KNN';
options.k = 5;
options.WeightMode = 'Binary';
W = constructW(X',options);
D = diag(sum(W,1));
L=D-W;

% This part of code initialize the F and G of the data matrix
idx_G=kmeans(X',K,'Emptyaction','drop');
G=zeros(n,K);
for j=1:length(idx_G)
    G(j,idx_G(j))=1;
end
G=G+0.2;   % numerical stability purpose
    
idx_F=kmeans(X,K,'Emptyaction','drop');
F=zeros(p,K);
for i=1:length(idx_F)
    F(i,idx_F(i))=1;
end
F=F+0.2;

% This par of code initialize the gamma and sigma 

eps=1e-4;
gamma=zeros(p,n);
sigma=zeros(size(G));
maxiter=1000;
k=1;   % counting number of iterations
G;
while k<maxiter
    % This part minimizes E
    temp1=X-F*G'+1/mu*gamma;
    [E]=L21_solver(temp1,1/mu);
    
    % This part minimizes F
    F=(X-E+1/mu*gamma)*G*(G'*G)^(-1);
    
    % This part minimizes H
    temp2=G+1/mu*sigma-lambda/mu*(G'*L)';
    [H]=nonneg_L2(temp2);
    
    % This part minimizes G
    temp3=H-1/mu*sigma+lambda/mu*L*H+(F'*(X-E+1/mu*gamma))';
    [U,S,V]=svd(temp3,0);
    G=U*V';
    
    % update the parameters
    gamma=gamma+mu*(X-F*G'-E);
    sigma=sigma+mu*(G-H);
    mu=pho*mu;
    
    k=k+1;
end
k;


for i=1:size(G,1)
    [temp1,temp2]=max(G(i,:));
    idx(i)=temp2;
end
[result,confus] = ClusteringMeasure(Y, idx);
result
end