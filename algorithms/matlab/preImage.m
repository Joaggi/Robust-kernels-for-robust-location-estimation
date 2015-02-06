function Z = preImage(X, KK, H, s, maxepocs)
    D = diag(KK)*ones(1,size(H,2)) + ones(size(X,2),1)*diag(H'*KK*H)' - KK*H;
    [a I] = min(D);
    clear a;
    Z = X(:,I); 
    while maxepocs>0
        K = gKernel(X,Z,s).*H;
        Z = X*K*diag(1./(1e-300+sum(K)));
        maxepocs = maxepocs - 1;
    end
