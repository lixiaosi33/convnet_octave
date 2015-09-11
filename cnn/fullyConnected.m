function fullyConnectedActivations = fullyConnected(filterDim, numFilters, numImages, activations, W, b)
%cnnConvolve Returns the convolution of the features given by W and b with
%the given images
%
% Parameters:
%  filterDim - filter (feature) dimension
%  numFilters - number of feature maps
%  numImages - number of images
%  activations -  activation matrix in the form
%           images(r, c, filterno, imageno)
%  W, b - W, b for features from the sparse autoencoder
%         W is of shape (filterDim,filterDim,numFilters)
%         b is of shape (numFilters,1)
%
% Returns:
%  fullyConnectedActivations - matrix of  activations in the form
%                      fullyConnectedActivations(numClasses, numImages)

numClasses = size(W, 1);

activationsPooled = reshape(activations,[],numImages);

fullyConnectedActivations = zeros(numClasses,numImages);

for imageNum = 1:numImages 
    activation  = activationsPooled(:, imageNum);
    output = W * activation;
    
    exp_output = exp(output);
    softmax_output = exp_output/sum(exp_output);
    
    fullyConnectedActivations(:, imageNum) = softmax_output;
end


end

