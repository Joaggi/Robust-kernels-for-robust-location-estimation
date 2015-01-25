load('dat');
dat;
k = 20;
addpath 'G:\Dropbox\Universidad\Machine Learning\Algorithms\Matlab'
X = dat(:,1:k*100);
X= L1norm(X);
X = X';

[F,G] = set_variables(X,k);
G = G';
X = X';
fprintf('Legue')
[idx] = L21NMF(X,F,G,0.01);

vectorY = zeros(size(idx));
for i =1:k
vectorY(1+(i-1)*100:100*i)= i;
end

[purity,purityVector] = calculate_purity(idx,vectorY,k);