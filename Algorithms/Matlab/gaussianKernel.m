function K = gaussianKernel(X, s)
    K = exp(-EuDist2(X).^2/s);
end