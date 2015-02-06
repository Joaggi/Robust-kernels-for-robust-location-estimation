data_contamination = zeros(size(data_real))
for i =1:213
    image = reshape(data_real(i,:),32,32);
    if randi(2) == 1
        if randi(2) == 1
            image(1:16,:)=0;

        else 
            image(17:32,:)=0;
        end
    else
        if randi(2) == 1
            image(:,1:16)=0;
        else 
            image(:,17:32)=0;
        end    
    end
    data_contamination(i,:) = image(:);
end