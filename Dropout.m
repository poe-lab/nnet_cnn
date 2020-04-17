classdef Dropout < nnet.internal.cnn.layer.Layer
    % Dropout   Implementation of the dropout layer
    
    %   Copyright 2015-2016 The MathWorks, Inc.

    properties
        % LearnableParameters   Learnable parameters for the layer
        %   This layer has no learnable parameters.
        LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();

        % Name (char array)   A name for the layer
        Name
    end
    
    properties (Constant)
        % DefaultName   Default layer's name.
        DefaultName = 'dropout'
    end
            
    properties (SetAccess = private)
        % HasSizeDetermined   Specifies if all size parameters are set
        %   For this layer, there are no size parameters to set.
        HasSizeDetermined = true
        
        % Fraction   The proportion of neurons to drop
        %   A number between 0 and 1 which specifies the proportion of
        %   input elements that are dropped by the dropout layer.
        Probability
    end
    
    methods
        function this = Dropout(name, probability)
            this.Name = name;
            this.Probability = probability;
        end
        
        function [Z, dropoutMask] = forward(this, X)
            % Use "inverted dropout", where we use scaling at training time
            % so that we don't have to scale at test time. The scaled
            % dropout mask is returned as the variable "dropoutMask".
            dropoutScaleFactor = cast( 1 - this.Probability, superiorfloat(X) );
            dropoutMask = ( rand(size(X), 'like', X) > this.Probability ) / dropoutScaleFactor;
            Z = X.*dropoutMask;
        end
        
        function gradients = gradients( ~, ~, ~ )
            % gradients    No-op since this layer does not contain any
            % learnable parameters
            gradients = [];
        end     
        
        function dX = backward(~, ~, ~, dZ, mask)
            dX = dZ.*mask;
        end
        
        function outputSize = forwardPropagateSize(~, inputSize)
            outputSize = inputSize;
        end
        
        function this = inferSize(this, ~)
        end
        
        function tf = isValidInputSize(~, ~)
            % isValidInputSize   Check if the layer can accept an input of
            % a certain size
            tf = true;
        end
        
        function this = initializeLearnableParameters(this, ~)
        end
        
        function this = prepareForTraining(this)
            this.LearnableParameters = nnet.internal.cnn.layer.learnable.TrainingLearnableParameter.empty();
        end
        
        function this = prepareForPrediction(this)
            this.LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();
        end
        
        function this = setupForHostPrediction(this)
        end
        
        function this = setupForGPUPrediction(this)
        end
    end
end