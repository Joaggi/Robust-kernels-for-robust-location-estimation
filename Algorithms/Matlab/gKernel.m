function K = gKernel(X, Y, s)
    if nargin == 2,
        K = diag(X)*ones(1,size(X,2))+ ones(size(X,2),1)*diag(X)'- 2*X;
	K = exp((-1/Y).*K);
    else
        K = diag(X'*X)*ones(1,size(Y,2))+ ones(size(X,2),1)*diag(Y'*Y)'- 2*X'*Y;
	K = exp((-1/s).*K);
    end
    
