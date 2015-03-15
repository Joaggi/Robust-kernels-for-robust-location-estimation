function Z = preW(X, C, s, maxepocs)
    [~,I] = max(C);
    Z = X(:,I); 
    while maxepocs>0
        K = gKernel(X,Z,s).*C;
        Z = X*(K)*diag(1./(1e-300 + sum(K)));
        maxepocs = maxepocs - 1;
    end
%     K = gKernel(X,Z,s).*C;
%     Y = (K)*diag(1./(1e-300 + sum(K)));
end
