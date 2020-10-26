
 categories = {'ImageNet/CIFAR10/Tiny-ImageNet Categories, comma separated'}; %this one has a DS of 2, Increase to 4 for large dataset usage
 rootFolder = 'Your FilePATH';
 
imds = imageDatastore(fullfile(rootFolder, categories),'LabelSource', 'foldernames');

k = 1; 
netWidth = 32;
Alpha_init = 6;
sigma1 = 0.5;
sigma2 = 1;
layers = [
    imageInputLayer(['your image dimensions],'Name','input')
    convolution2dLayer(3,netWidth,'Padding','same','Name','convInp')
    batchNormalizationLayer('Name','BNInp')
    %reluLayer('Name','reluInp')
    AAreluLayer(Alpha_init,'reluInp')
    
    convolutionalUnit_NoDown(netWidth,1,'S1U1')
    additionLayer(2,'Name','add11')
    %reluLayer('Name','relu11')
    AAreluLayer(Alpha_init,'relu11')    
    convolutionalUnit_NoDown(netWidth,1,'S1U2')
    additionLayer(2,'Name','add12')
    %reluLayer('Name','relu12')
    AAreluLayer(Alpha_init,'relu12')
    

    
    convolutionalUnit_Down(k*netWidth,2,'S2U1')%large
    
    additionLayer(2,'Name','add21')
    %reluLayer('Name','relu21')
    AAreluLayer(Alpha_init,'relu21')    
    convolutionalUnit_NoDown(k*netWidth,1,'S2U2')
    additionLayer(2,'Name','add22')
    %reluLayer('Name','relu22')
    AAreluLayer(Alpha_init,'relu22')
    

    
    convolutionalUnit_Down_2(2*k*netWidth,2,'S3U1')%small
    
    additionLayer(2,'Name','add31')
    %reluLayer('Name','relu31')
    AAreluLayer(Alpha_init,'relu31')
    
    convolutionalUnit_NoDown(2*k*netWidth,1,'S3U2')
    additionLayer(2,'Name','add32')
    %reluLayer('Name','relu32')
    AAreluLayer(Alpha_init,'relu32')
    
    averagePooling2dLayer(8,'Name','globalPool')
    fullyConnectedLayer(10,'Name','fcFinal')
    
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')
    ];

lgraph = layerGraph(layers);
% figure('Units','normalized','Position',[0.2 0.2 0.6 0.6]);
% plot(lgraph);
lgraph = connectLayers(lgraph,'reluInp','add11/in2');
lgraph = connectLayers(lgraph,'relu11','add12/in2');

% figure('Units','normalized','Position',[0.2 0.2 0.6 0.6]);
% plot(lgraph);

skip1 = [
        
    convolution2dLayer(1,k*netWidth,'Name','skipConv1')
    batchNormalizationLayer('Name','skipBN1')

        maxPooling2dLayer(2, 'Stride', 1,'Padding', 'same')      
        gauss_pool(sigma1,'bp23')
        maxPooling2dLayer(2, 'Stride', 2,'Padding', 'same','Name','pool23')];
        
        
lgraph = addLayers(lgraph,skip1);
lgraph = connectLayers(lgraph,'relu12','skipConv1');%
lgraph = connectLayers(lgraph,'pool23','add21/in2');

lgraph = connectLayers(lgraph,'relu21','add22/in2');

skip2 = [
        
    convolution2dLayer(1,2*k*netWidth,'Name','skipConv2')
    batchNormalizationLayer('Name','skipBN2')
        maxPooling2dLayer(2, 'Stride', 1,'Padding', 'same')      
        gauss_pool(sigma2,'bp24')
        maxPooling2dLayer(2, 'Stride', 2,'Padding', 'same','Name','pool22')];
        
lgraph = addLayers(lgraph,skip2);
lgraph = connectLayers(lgraph,'relu22','skipConv2');%
lgraph = connectLayers(lgraph,'pool22','add31/in2');

lgraph = connectLayers(lgraph,'relu31','add32/in2');

numUnits = 9;
netWidth = 16;

miniBatchSize = 128;
learnRate = 0.1*miniBatchSize/128;
%valFrequency = floor(size(XTrain,4)/miniBatchSize)  'L2Regularization',.0005, ...
options = trainingOptions('sgdm', ...
    'InitialLearnRate',.1, ...
    'MaxEpochs',160, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'Plots','none', ...
    'Verbose',true, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',60);

 resNet= trainNetwork(imds, lgraph, options);

 rootFolder1 = 'Your FilePATH';
 imds_test = imageDatastore(fullfile(rootFolder1, categories),'LabelSource', 'foldernames');


labels = classify(resNet, imds_test);
confMat = my_confusionmat(imds_test.Labels, labels);
confMat = confMat./sum(confMat,2);
mean(diag(confMat))

function layers = convolutionalUnit_Down_2(numF,stride,tag)
layers = [
    convolution2dLayer(3,numF,'Padding','same','Name',[tag,'convA1'])
    batchNormalizationLayer('Name',[tag,'BNA1'])
    %reluLayer('Name',[tag,'reluA1'])
    AAreluLayer(Alpha_init,[tag,'relu2'])
    
        gauss_pool(sigma2,[tag,'bpA']); 
        
        maxPooling2dLayer(2, 'Stride', 2,'Padding', 'same','Name',[tag,'PoolA'])
        
    %AAreluLayer(Alpha_init,[tag,'relu1'])
    convolution2dLayer(3,numF,'Padding','same','Name',[tag,'convA2'])
    batchNormalizationLayer('Name',[tag,'BNA2'])];
end

function layers = convolutionalUnit_Down(numF,stride,tag) %large
layers = [
    convolution2dLayer(3,numF,'Padding','same','Name',[tag,'convA1'])
    batchNormalizationLayer('Name',[tag,'BNA1'])
    %reluLayer('Name',[tag,'reluA1'])
    AAreluLayer(Alpha_init,[tag,'relu1'])
        gauss_pool(sigma1,[tag,'bpA']); 
        
        maxPooling2dLayer(2, 'Stride', 2,'Padding', 'same','Name',[tag,'PoolA'])
        
    %AAreluLayer(Alpha_init,[tag,'relu1'])
    convolution2dLayer(3,numF,'Padding','same','Name',[tag,'convA2'])
    batchNormalizationLayer('Name',[tag,'BNA2'])];
end

function layers = convolutionalUnit_NoDown(numF,stride,tag)
layers = [
    convolution2dLayer(3,numF,'Padding','same','Name',[tag,'conv1'])
    batchNormalizationLayer('Name',[tag,'BN1'])
    %reluLayer('Name',[tag,'relu1'])
    AAreluLayer(Alpha_init,[tag,'relu1'])
    convolution2dLayer(3,numF,'Padding','same','Name',[tag,'conv2'])
    batchNormalizationLayer('Name',[tag,'BN2'])];
end