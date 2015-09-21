function [opttheta] = minFuncSGD(funObj,theta,data,labels,options, testImages, testLabels, numClasses, filterDim, numFilters, poolDim)
% Runs stochastic gradient descent with momentum to optimize the
% parameters for the given objective.
%
% Parameters:
%  funObj     -  function handle which accepts as input theta,
%                data, labels and returns cost and gradient w.r.t
%                to theta.
%  theta      -  unrolled parameter vector
%  data       -  stores data in m x n x numExamples tensor
%  labels     -  corresponding labels in numExamples x 1 vector
%  options    -  struct to store specific options for optimization
%
% Returns:
%  opttheta   -  optimized parameter vector
%
% Options (* required)
%  epochs*     - number of epochs through data
%  alpha*      - initial learning rate
%  minibatch*  - size of minibatch
%  momentum    - momentum constant, defualts to 0.9


%%======================================================================
%% Setup
assert(all(isfield(options,{'epochs','alpha','minibatch'})),...
        'Some options not defined');
if ~isfield(options,'momentum')
    options.momentum = 0.9;
end;
epochs = options.epochs;
alpha = options.alpha;
minibatch = options.minibatch;
m = length(labels); % training set size
% Setup for momentum
mom = 0.5;
momIncrease = 20;
velocity = zeros(size(theta));
look_ahead=zeros(size(theta));
%%======================================================================
%% SGD loop
it = 0;
for e = 1:epochs
    
    % randomly permute indices of data for quick minibatch sampling
    rp = randperm(m);
    
    for s=1:minibatch:(m-minibatch+1)
        printf('Progress: %f/%f, %f/%f', e, epochs, s, m-minibatch+1)
        it = it + 1;

        % increase momentum after momIncrease iterations
        if it == momIncrease
            mom = options.momentum;
        end;

        % get next randomly selected minibatch
        mb_data = data(:,:,rp(s:s+minibatch-1));
        mb_labels = labels(rp(s:s+minibatch-1));

        % evaluate the objective function on the next minibatch
        [cost grad] = funObj(theta,mb_data,mb_labels);
        % Instructions: Add in the weighted velocity vector to the
        % gradient evaluated above scaled by the learning rate.
        % Then update the current weights theta according to the
        % sgd update rule
        
        %%% YOUR CODE HERE %%%
        look_ahead = theta+mom*velocity;
        [cost2 grad_ahead] = funObj(look_ahead,mb_data,mb_labels);
        velocity = mom*velocity - alpha*grad_ahead;
        theta =theta+velocity;
        
        fprintf('Epoch %d: Cost on iteration %d is %f\n',e,it,cost);

        dlmwrite(options.log_file_name, [it, cost], '-append')

        if cost < 0.4
            fprintf('Evaluating')

            % Acuuracy Evaluation
            [~,cost,preds]=cnnCost(theta,testImages,testLabels,numClasses,filterDim,numFilters,poolDim,true);

            acc = sum(preds==testLabels)/length(preds);

            % Accuracy should be around 97.4% after 3 epochs
            fprintf('Accuracy is %f\n',acc);
        end
    end;

    % aneal learning rate by factor of two after each epoch
    alpha = alpha/2.0;

end;

opttheta = theta;

end
