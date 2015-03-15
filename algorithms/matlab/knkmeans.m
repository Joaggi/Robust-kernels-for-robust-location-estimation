function [label, energy] = knkmeans(K,k,option,X)
% Perform kernel k-means clustering.
%   K: kernel matrix
%   init: k (1 x 1) or label (1 x n, 1<=label(i)<=k)
% Reference: [1] Kernel Methods for Pattern Analysis
% by John Shawe-Taylor, Nello Cristianini
% Written by Michael Chen (sth4nth@gmail.com).
n = size(K,1);
K = double(K);

switch option.initialization
    case 'kmeans'
        label=kmeans(X',k,'Emptyaction','drop'); % k-mean clustering, get idx=[1,1,2,2,1,3,3,1,2,3]
    case 'random'
        label = ceil(k*rand(1,n))';
    case 'kkmeans'
        label = ceil(k*rand(1,n))';
end



last = 0;
contador = 0
while contador <=10
    [u,~,label] = unique(label);   % remove empty clusters
    k = length(u);
    E = sparse(label,1:n,1,k,n,n);
    E = bsxfun(@times,E,1./sum(E,2));
    T = E*K;
    Z = repmat(diag(T*E'),1,n)-2*T;
    last = label;
    [val, label] = min(Z,[],1);
    if any(label(:) ~= last(:))
        contador = 0;  
    else
        contador = contador+1;   
    end
end
[~,~,label] = unique(label);   % remove empty clusters
energy = sum(val)+trace(K);