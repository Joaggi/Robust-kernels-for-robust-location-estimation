function [E]=L21_solver(Q,lambda)
% this function solves the equation
%  min 1/2||W-Q||_F^2+lambda*||W||_2,1

E=zeros(size(Q));
for i=1:size(Q,2)
    temp=norm(Q(:,i),2);
    if temp<lambda
        E(:,i)=0;
    else E(:,i)=(temp-lambda)/temp*Q(:,i);
    end
end
