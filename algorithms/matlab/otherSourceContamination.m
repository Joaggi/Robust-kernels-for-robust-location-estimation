function [Y per] = otherSourceContamination(X,alpha,filename)
load(filename, 'con');
per = ceil(rand(1,size(X,2)*alpha)*size(con,2));
cond = con(:, per);
Y = [X cond];
