function [F,G] = set_variables(X,k)

idx = kmeans(X,k);
G = zeros(size(idx,1),k);
for i=1:k
    G(find(idx==i),i) = 1;
end
F = X'*G;
end

