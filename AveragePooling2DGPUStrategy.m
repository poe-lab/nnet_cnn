classdef AveragePooling2DGPUStrategy < nnet.internal.cnn.layer.util.ExecutionStrategy
    % AveragePooling2DGPUStrategy   Execution strategy for running the average pooling on the GPU
    
    %   Copyright 2016 The MathWorks, Inc.
    
    methods
        function [Z, memory] = forward(~, X, ...
                poolHeight, poolWidth, ...
                verticalPad, horizontalPad, ...
                verticalStride, horizontalStride)
            Z = nnet.internal.cnngpu.poolingAverageForward2D(X, ...
                poolHeight, poolWidth, ...
                verticalPad, horizontalPad, ...
                verticalStride, horizontalStride);
            memory = [];
        end
    end
end