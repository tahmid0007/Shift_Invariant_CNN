classdef gauss_pool < nnet.layer.Layer
    
    properties   
        Sigma
    end
    
    methods
        function layer = gauss_pool(sig, name)
            
            layer.Name = name;
            layer.Sigma = sig;
            layer.Description = "Gaussian pooling " + sig + " Sigma";
            
        end
        
        function Z = predict(layer, X)
            Z= imgaussfilt(X,layer.Sigma,'FilterSize',3,'FilterDomain','spatial');
        end
        
        function [dLdX] = backward(layer, X, Z, dLdZ, memory)
            H = fspecial('gaussian',3,layer.Sigma);
            [h w c b] = size(dLdZ);
            
            if b > 1
                for i=1:c
                    
                    dLdX(:,:,i,:) = imfilter(dLdZ(:,:,i,:),H);
                    
                end
            else
                for i=1:c
                    dLdX(:,:,i) = imfilter(dLdZ(:,:,i),H);
                end
            end
            
        end
    end
end
