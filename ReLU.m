classdef ReLU < nnet.internal.cnn.layer.Layer
    % ReLU   Rectified Linear Unit layer
    
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
        DefaultName = 'relu'
    end
            
    properties (SetAccess = private)
        % HasSizeDetermined   True for layers with size determined.
        HasSizeDetermined = true
    end
    
    properties(Access = private)
        ExecutionStrategy
    end
    
    methods
        function this = ReLU(name)
            % ReLU  Constructor for the layer
            this.Name = name;
            
            this.ExecutionStrategy = nnet.internal.cnn.layer.util.ReLUGPUStrategy();
        end
        
        function [Z, memory] = forward( this, X )
            % forward   Forward input data through the layer and output the result
            [Z, memory] = this.ExecutionStrategy.forward(X);
        end
        
        function dX = backward( ~, X, Z, dZ, ~ )
            % backward    Back propagate the derivative of the loss function
            % thru the layer
            dX = nnet.internal.cnngpu.reluBackward(Z, dZ, X);
        end
        
        function gradients = gradients( ~, ~, ~ )
            % gradients    No-op since this layer does not contain any
            % learnable parameters
            gradients = [];
        end        
        
        function outputSize = forwardPropagateSize(~, inputSize)
            % forwardPropagateSize  Output the size of the layer based on
            %                       the input size
            outputSize = inputSize;
        end
        
        function this = inferSize(this, ~)
            % inferSize     no-op since this layer has nothing that can be
            %               inferred
        end
        
        function tf = isValidInputSize(~, ~)
            % isValidInputSize   Check if the layer can accept an input of
            % a certain size
            tf = true;
        end
        
        function this = initializeLearnableParameters(this, ~)
            % initializeLearnableParameters     no-op since there are no
            %                                   learnable parameters
        end
        
        function this = prepareForTraining(this)
            this.LearnableParameters = nnet.internal.cnn.layer.learnable.TrainingLearnableParameter.empty();
        end
        
        function this = prepareForPrediction(this)
            this.LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();
        end
        
        function this = setupForHostPrediction(this)
            this.ExecutionStrategy = nnet.internal.cnn.layer.util.ReLUHostStrategy();
        end
        
        function this = setupForGPUPrediction(this)
            this.ExecutionStrategy = nnet.internal.cnn.layer.util.ReLUGPUStrategy();
        end
    end
end
