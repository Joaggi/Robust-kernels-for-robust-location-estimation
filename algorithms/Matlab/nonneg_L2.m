function [H]=nonneg_L2(temp)
% This function returns the max of temp elements and 0.

[n,p]=size(temp);
for i=1:n
    for j=1:p
        H(i,j)=max(temp(i,j),0);
    end
end