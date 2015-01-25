function D = dis(F,FC)
    D = diag(F'*F)*ones(1,size(FC,2))+ones(size(F,2),1)*diag(FC'*FC)'-2*F'*FC;
