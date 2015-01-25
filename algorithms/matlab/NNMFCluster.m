function [indCluster,Xout,Aout,Yout]=NNMFCluster(X,k)
% NNMF based clustering
%%%%

callNMF=true;
if nargin==2
    option=[];
    callNMF=true;
end
if nargin==1
    Y=X; % X is coefficient matrix
    A=[];
    callNMF=false;
    option=[];
end


if nargin==1
    Y=X; % X is coefficient matrix
    callNMF=false;
end
if callNMF
   [A,Y]=nnmf(X,k);
end
[C,indCluster]=max(Y,[],1);
indCluster=indCluster';
Xout=[];
Aout=[];
Yout=[];
    [indClusterSorted,ind]=sort(indCluster);
    Xout=X(:,ind);
    Yout=Y(:,ind);
    Aout=A;
end