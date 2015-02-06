function gramMatrix = andrewsKernel(X,Y, c)
%     gramMatrix = zeros(size(X,2),size(X,2));
%     K = sqrt(diag(X'*X)*ones(1,size(Y,2))+ ones(size(X,2),1)*diag(Y'*Y)'- 2*X'*Y);
% 	g = (1/(2*pi^2).*cos(pi*(1/c).*K));
%     gramMatrix(K/c <= 1)= g(K/c <= 1);
%     gramMatrix(K/c > 1) = -1.0/(2.0*pi^2)
    
    gramMatrix = zeros(size(X,2),size(X,2));
    K = sqrt(diag(X'*X)*ones(1,size(Y,2))+ ones(size(X,2),1)*diag(Y'*Y)'- 2*X'*Y);
	g = (1-((1/c).*K).^2).^3;
    gramMatrix(K/c <= 1)= g(K./c <= 1);
    gramMatrix(K/c > 1) = 0;
end
    
