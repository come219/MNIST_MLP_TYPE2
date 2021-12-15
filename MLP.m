%   __________________________
%   COMP6011: Machine Learning
%   19129576
%   MLP MNIST CW
%   _____________________________



% Note: this file merely specifies the MLP class. It is not meant to be
% executed as a stand-alone script. The MLP needs to be instantiated and
% then used elsewhere, see e.g. 'testMLP131train.m'.

% A Multi-layer perceptron class
classdef MLP < handle
    % Member data
    properties (SetAccess=private)
        inputDimension % Number of inputs
        hiddenDimension % Number of hidden neurons
        outputDimension % Number of outputs
        
        hiddenLayerWeights % Weight matrix for the hidden layer, format (hiddenDim)x(inputDim+1) to include bias terms
        outputLayerWeights % Weight matrix for the output layer, format (outputDim)x(hiddenDim+1) to include bias terms

    end
    
    methods
        % Constructor: Initialize to given dimensions and set all weights
        % zero.
        % inputD ~ dimensionality of input vectors
        % hiddenD ~ number of neurons (dimensionality) in the hidden layer 
        % outputD ~ number of neurons (dimensionality) in the output layer 
        function mlp=MLP(inputD,hiddenD,outputD)
            mlp.inputDimension=inputD;
            mlp.hiddenDimension=hiddenD;
            mlp.outputDimension=outputD;
            mlp.hiddenLayerWeights=zeros(hiddenD,inputD+1);
            mlp.outputLayerWeights=zeros(outputD,hiddenD+1);
            
          
            
        end
        
        % TODO Implement a randomized initialization of the weight
        % matrices.
        % Use the 'stdDev' parameter to control the spread of initial
        % values.
        function mlp=initializeWeightsRandomly(mlp,stdDev)
            % Note: 'mlp' here takes the role of 'this' (Java/C++) or
            % 'self' (Python), refering to the object instance this member
            % function is run on.
            
            
            % use zeroes function which creates a scalar on the
            % corresponding layers (zeros
            
             mlp.hiddenLayerWeights=zeros(mlp.hiddenDimension,mlp.inputDimension+stdDev);% TODO
             mlp.outputLayerWeights=zeros(mlp.outputDimension,mlp.hiddenDimension+stdDev);% TODO
            
            %initialize a set of weights for the hidden and output layer
             mlp.hiddenLayerWeights = rand(mlp.hiddenDimension, mlp.inputDimension);
             mlp.outputLayerWeights = rand(mlp.outputDimension, mlp.hiddenDimension);
            
            
            
            
             %create new set of hidden weights using averages
            mlp.hiddenLayerWeights = mlp.hiddenLayerWeights./size( mlp.hiddenLayerWeights, 2);
            
            %create new set of output weights using averages
            mlp.outputLayerWeights = mlp.outputLayerWeights./size( mlp.outputLayerWeights, 2); 
            
        end
        
        % TODO Implement the forward-propagation of values algorithm in
        % this method.
        % 
        % inputData ~ a vector of data representing a single input to the
        % network in column format. It's dimension must fit the input
        % dimension specified in the contructor.
        % 
        % hidden ~ output of the hidden-layer neurons
        % output ~ output of the output-layer neurons
        % 
        % Note: the return value is automatically fit into a array
        % containing the above two elements
        function [hidden,output]=compute_forward_activation(mlp, inputData)
                      
            %number of training vectors
            trainingSetSize = size(inputData, 2);
            
            % Choose activation function, logistic sigmoid
            activationFunction = @logisticSigmoid;
            
            %assign the scope value for the batch size
            batchSize = 100;
            n = zeros(batchSize);
           
            for k = 1 : batchSize
            
                %hidden = zeros(mlp.hiddenDimension);% TODO
                %output = zeros(mlp.outputDimension);% TODO
                %select which input vector to train
                n(k) = floor(rand(1)*trainingSetSize + 1);
                
                inputVector = inputData(:, n(k));

                %mlp.hiddenDimension = mlp.hiddenLayerWeights * hiddenVector;
                %creates a new hidden dimension using the hidden weights
                %with input. returns a vector using the activiation func
                mlp.hiddenDimension  = mlp.hiddenLayerWeights * inputVector;
                hidden = activationFunction(mlp.hiddenDimension);
         
                %mlp.outputDimension = mlp.outputLayerWeights.* hiddenOutputVector;
                %creates a new output dimension using the output weights
                %and hidden vector
                mlp.outputDimension  = mlp.outputLayerWeights.* hidden;
                output = activationFunction(mlp.outputDimension);
            end
  
        end
        
        
        % This function calls the forward propagation and extracts only the
        % overall output. It does not have to be altered.
        function output=compute_output(mlp,input)
            [~,output] = mlp.compute_forward_activation(input);
            
        end
        
        
        % TODO Implement the backward-propagation of errors (learning) algorithm in
        % this method.
        %
        % This method implements MLP learning by means on backpropagation
        % of errors on a single data point.
        %
        % inputData ~  a vector of data representing a single input to the
        %   network in column format.
        % targetOutputData ~ a vector of data representing the
        %   desired/correct output the network should generate for the given
        %   input (this is the supervision signal for learning)
        % learningRate ~ step width for gradient descent
        %
        % This method is expected to update mlp.hiddenLayerWeights and
        % mlp.outputLayerWeights.
        function mlp=train_single_data(mlp, inputData, targetOutputData, learningRate)
            
            %assign epochs to this scope
            epoch = 1;
            
            % Transform the labels to correct target values.
            targetValues = 0.*ones(10, size(targetOutputData, 1));
            for n = 1: size(targetOutputData, 1)
                    targetValues(targetOutputData(n) + 1, n) = 1;
            end;
            
            trainingSetSize = size(inputData, 2);
            batchSize = 100;
            n = zeros(batchSize);
            
            
            for k = 1: batchSize
                    
                % select which input vector to train on
                n(k) = floor(rand(1)*trainingSetSize + 1);
                
                inputVector = inputData(:, n(k)); 
                
             %   _____
             %   function to compute forward prop
             %   compute forward propogationg
             %   returns a vector or output and input
             %   ______
                [h,o] = mlp.compute_forward_activation(inputData);
            
                %supervisor
                targetVector = targetOutputData(:,n(k));
            
                %apply the function to use logistic sigmoid
                dActivationFunction = @dLogisticSigmoid;
            
             %   _________
             %find deltas of output and hidden
             %   gradient descent for updating the weights
             %   _______
                
                %output delta uses the activation function of the actual
                %output multipled with the matrix of the output vector
                %minus the target
                outputDelta = dActivationFunction(mlp.outputDimension).*( o - targetVector );
                
                %hidden delta uses the activation function of the actual
                %hiddent multipled with the matrix of the output weights
                %matrix transpose multiplication with the outputdelta
                hiddenDelta = dActivationFunction(mlp.hiddenDimension).*(mlp.outputLayerWeights'.*outputDelta);
            
            %  _________
            % apply back prop
            %   _______
            
                %  new output layer weights is current minus learning rate matrix multipled with the output delta to the
                %  transpose of the hidden vector
                mlp.outputLayerWeights = mlp.outputLayerWeights - learningRate.*outputDelta.*h';
            
                %   the hidden weights becomes the current minus the learning
                %   rate multiplied by the hidden delta and then the transpose
                %   of the input vector
                mlp.hiddenLayerWeights = mlp.hiddenLayerWeights - learningRate.*hiddenDelta*inputVector';
            end;   
           
            
            
            %   ___
            %   error code
            % incorrect implementation,??
            %   _____

            %choose activation function
            activationFunction = @logisticSigmoid;
          
            % Calculate the error for plotting.
            error = 0;
            
            for k = 1: batchSize
                %select input vector
                n(k) = floor(rand(1)*trainingSetSize + 1);
                
                inputVector = inputData(:, n(k));
                targetVector = targetValues(:, n(k));
                %error = error + norm(activationFunction(mlp.outputLayerWeights.*activationFunction(mlp.hiddenLayerWeights.*inputVector)) - targetVector, 2);
            end;
            
            %fprintf('error: %d', error);
            error = error/batchSize;
            
            plot(epoch, error,'*');
   
            %add epoch to epoch to correspond to loop in main
            epoch = epoch + 1;
        
        end;
        
        
    end
    
end
