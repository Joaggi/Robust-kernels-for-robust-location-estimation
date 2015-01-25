function X = standarization(X)
   X = (X - repmat(mean(X),size(X,1),1)) ./ repmat(std(X),size(X,1),1)
