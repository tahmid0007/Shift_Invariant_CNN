%%%%%%%%%%This is AA RELU%%%%%%%%%
classdef AAreluLayer < nnet.layer.Layer
    properties (Learnable)
        Alpha
    end
    methods
        function layer = AAreluLayer(Threshold, name)
            
            layer.Name = name;
            
            layer.Description = "AAreluLayer with " + Threshold + " as Alpha";
            
            layer.Alpha = Threshold; %Typically 6
        end
        
        function Z = predict(layer, X)
            
            %Z =   (0 < X & X <= layer.Alpha).*X + ((layer.Alpha < X).*(layer.Alpha.*sin(log(abs(X+.01)./layer.Alpha))+layer.Alpha)); %oscilate
			 Z =   (0 < X & X <= layer.Alpha).*X + ((layer.Alpha < X & X < 28.86).*(layer.Alpha.*sin(log(abs(X+.01)./layer.Alpha))+layer.Alpha))+(28.86 <= X).*12; %28.86 is the max (alpha = 6) where x is clipped.       
        end
        
    end
end