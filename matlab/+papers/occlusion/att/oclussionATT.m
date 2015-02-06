for i =1:400
    image = reshape(data(i,:),23,19)
    if randi(2) == 1
        if randi(2) == 1
            image(1:12,:)=0

        else 
            image(12:23,:)=0
        end
    else
        if randi(2) == 1
            image(:,1:10)=0
        else 
            image(:,10:19)=0
        end    
    end
    data(i,:) = image(:)
end