function gramMatrix = gemanKernel(X,Y, c)
    
    gramMatrix = zeros(size(X,2),size(X,2));
    K = diag(X'*X)*ones(1,size(Y,2))+ ones(size(X,2),1)*diag(Y'*Y)'- 2*X'*Y;
	gramMatrix = 2*K/(1+K);
end
    
