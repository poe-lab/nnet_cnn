classdef AveragePooling2DHostStrategy < nnet.internal.cnn.layer.util.ExecutionStrategy
    % AveragePooling2DHostStrategy   Execution strategy for running the average pooling on the host
    
    %   Copyright 2016 The MathWorks, Inc.
    
    methods
        function [Z, memory] = forward(~, X, ...
                poolHeight, poolWidth, ...
                verticalPad, horizontalPad, ...
                verticalStride, horizontalStride)
            Z = nnet.internal.cnnhost.poolingAverageForward2D(X, ...
                poolHeight, poolWidth, ...
                verticalPad, horizontalPad, ...
                verticalStride, horizontalStride);
            memory = [];
        end
    end
end