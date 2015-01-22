function [idx]=L21NMF(X,F,G,lambda)
% % this function implements the method described in the TKDD journal paper 
%  "Robust Manifold Non-Negative Matrix Factorization" by Jin Huang,
%   Feiping Nie, Heng Huang and Chris Ding. 
%  Please contact Jin Huang (huangjinsuzhou@gmail.com) if you had any question or 
%  found any bug in the code. Thanks!

% Input:
% X is row-based data matrix of size n by p
% F, G are initial matrix decomposition of X 
% lambda is the regularity parameter

% Output:
% n by 1 vector indicating memberships of rows in X


% In this code, we use affinity graph construction code from others, please go
% to http://www.cad.zju.edu.cn/home/dengcai/Data/DimensionReduction.html
% to download constructW function and understand the parameter setting here
% you can also construct the affinity graph in your own way

options = [];
options.Metric = 'Euclidean';
options.NeighborMode = 'KNN';
options.k = 5;    % number of neighbors
options.WeightMode = 'HeatKernel';
options.t = 1;
W = constructW(X',options);



n=size(X,2);
d = zeros(1,n);
for i = 1:n
    deno = norm(X(:,i)- F * G(:,i));
    d(i) = 1/deno;
end
D =diag(d);


D_g=diag(sum(W,2));



% construct the positive part and negative part of the residue matrix
eta=G*D*(X'-G'*F')*F-lambda*G*W*G';
eta=1/2*(eta+eta');

[eta_pos,eta_neg]=matrix_posneg(eta);



% iterative optimization
J = [];  % obj function value, make sure function value decreasing 
obj = L21f(X - F * G,W,G,lambda);
J = [J obj];
iter = 0;
while iter<=10000
    objO = obj;
    temp = G' .* ((D*X'*F+lambda*W*G'+D_g*G'*eta_neg) ./ (D*G'*F'*F+D_g*G'*eta_pos+eps));
    G=temp';
    F = F .* ((X * D * G') ./ (F * G * D * G'+eps));
    obj= L21f(X - F * G,W,G,lambda);
    J = [J obj];
    if abs(objO -obj)< 1e-3
        break;
    end
    D1 = X - F*G;
    D = diag (1 ./ sum( D1 .^2 , 1));
    eta=-G*D*(X'-G'*F')*F+lambda*G*W*G';
    eta=1/2*(eta+eta');
    [eta_pos,eta_neg]=matrix_posneg(eta);
    
    iter = iter +1;
end



for i=1:size(G,2)
    [temp1,temp2]=max(G(:,i));
    idx(i)=temp2;
end


end


function obj=L21f(X,W,G,lambda)
% calulate objective function value
d = sum((X .^ 2),1);
d1 = sqrt(d);
sum21=sum(d1);
obj=sum21-lambda/2*trace(G*W*G');
end

function [A_pos,A_neg]=matrix_posneg(A)
% this function calculates the positive A and negative A 
% such that A= A_pos - A_neg
[m,n]=size(A);
A_pos=zeros(m,n);
A_neg=zeros(m,n);

A_pos=(abs(A)+A)/2;
A_neg=(abs(A)-A)/2;
end