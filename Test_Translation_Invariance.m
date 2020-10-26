%5 pixel diagonal translate with black bag, flip prob % use this one for adversarial attacks as well, modify according to Engstorm et al.

count = 0;
M_sum = 0;
CT = 0;
n = 10000;
m = 8;
for j=1:n
    
    I(:,:,:,j) = imread(string(imds_test.Files(j)));
    [c1(:,j) s1(:,j)] = classify(resNet,I(:,:,:,j));
    
    for i =1:m
              
        J = imtranslate(I(:,:,:,j),[i, i]);
        c2 = classify(resNet,J);
        
        %[max1 ind1] = max(s1(:,i));
        
        if c1(:,j) ~= c2
            count = count + 1;
        end
        

    end
        CT = CT + count;
        count = 0;
end
MM = CT/(n*m)



%1 pixel diagonal translate with black bag

count = 0;
M_sum = 0;
n = 10000;
for i=1:n
    
    I = imread(string(imds_test.Files(i)));
    [c1(:,i) s1(:,i)] = classify(resNet,I);
    
    J = imtranslate(I,[1, 1]);
    [c2(:,i) s2(:,i)] = classify(resNet,J);
    
    [max1 ind1] = max(s1(:,i));
    
    if c1(:,i) ~= c2(:,i)
        count = count + 1;
    end
    temp = abs(max1 - s2(ind1,i));
    M_sum = M_sum + temp;
    
end
P1 = count/n
M1 = M_sum/n

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% resize original image and embed into a black background, than translate 1
% pixel diagonally

count = 0;
M_sum = 0;
n = 10000;
for i=1:n
    
    I = imread(string(imds_test.Files(i)));
    
    J_bg = uint8(zeros([32 32 3]));
    J_scaled = imresize(I, [24 24]);
    J_bg(4:27,4:27,:) = J_scaled;
    
    [c1(:,i) s1(:,i)] = classify(resNet,J_bg);
    
    J = imtranslate(J_bg,[1, 1]);
    [c2(:,i) s2(:,i)] = classify(resNet,J);
    
    [max1 ind1] = max(s1(:,i));
    
    if c1(:,i) ~= c2(:,i)
        count = count + 1;
    end
    temp = abs(max1 - s2(ind1,i));
    M_sum = M_sum + temp;
    
end
P2 = count/n
M2 = M_sum/n


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% resize original image and embed into a black background, than resize the
% embedded crop by 1 pixel 

count = 0;
M_sum = 0;
n = 10000;
for i=1:n
    
    I = imread(string(imds_test.Files(i)));
    
    J_bg1 = uint8(zeros([32 32 3]));
    J_scaled1 = imresize(I, [24 24]);
    J_bg1(4:27,4:27,:) = J_scaled1;
    
    [c1(:,i) s1(:,i)] = classify(resNet,J_bg1);
    
    J_bg2 = uint8(zeros([32 32 3]));
    J_scaled2 = imresize(I, [25 25]);
    J_bg2(4:28,4:28,:) = J_scaled2;
    
    [c2(:,i) s2(:,i)] = classify(resNet,J_bg2);
    
    [max1 ind1] = max(s1(:,i));
    
    if c1(:,i) ~= c2(:,i)
        count = count + 1;
    end
    temp = abs(max1 - s2(ind1,i));
    M_sum = M_sum + temp;
    
end
P3 = count/n
M3 = M_sum/n
