function [deltas_Wc, deltas_bc, deltas_Wd, deltas_bd] = backpropogate(imageDim, poolDim, images, activationsConv, activationsPooled, activationsSoftmax, labels, Wc, bc, Wd, bd)
%cnnConvolve Returns the convolution of the features given by W and b with
%the given images
%
% Parameters:
%  activationsSoftmax
%  labels
%
% Returns:
%  cost
%

numImages = size(activationsSoftmax, 2);
numClasses = size(activationsSoftmax, 1);
numFilters = size(Wc, 3);
filterDim = size(Wc, 1);
convDim = imageDim - filterDim + 1;
poolLayerDim = convDim / poolDim;

y = expandLabels(labels, activationsSoftmax);

sigmoid_prime = activationsConv.*(1-activationsConv);

deltas_fc = activationsSoftmax - y;

deltas_conv = zeros(convDim, convDim, numFilters);
deltas_pooling = zeros(poolLayerDim, poolLayerDim, numFilters);

deltas_Wc = zeros(size(Wc));
deltas_bc = zeros(size(bc));

deltas_Wd = zeros(size(Wd));
deltas_bd = zeros(size(bd));

for i = 1:numImages
% backpropogate deltas from fc softmax layer to pooling layer
  output = deltas_fc(:, i);
  
  error_pooling = reshape(Wd'*output, poolLayerDim, poolLayerDim, numFilters);
  deltas_pooling = deltas_pooling + error_pooling;
  
  aPooledFlat = reshape(activationsPooled,[],numImages);
  
  deltas_Wd = deltas_Wd + (output*aPooledFlat(:,i)');
  deltas_bd = deltas_bd + deltas_fc(:, i);
  
  for j = 1:numFilters
    % Backprop from pool to conv
    sigmoid_prime = activationsConv.*(1-activationsConv);
    error_conv = kron(error_pooling(:,:,j), ones(poolDim)).*sigmoid_prime(:,:,j,i);
    error_conv = error_conv/poolDim^2;
    deltas_conv(:,:,j) = deltas_conv(:,:,j) + error_conv;
    
    % Backpropogate from conv to input layer
    error = rot90(error_conv, 2);

    deltas_Wc(:, :, j) = deltas_Wc(:,:,j) + conv2(images(:,:,i), error, 'valid');
    deltas_bc(j) = deltas_bc(j) + sum(sum(error_conv));
   end
  end

  deltas_conv = deltas_conv / numImages;
  deltas_Wc = deltas_Wc / numImages;
  deltas_bc = deltas_bc / numImages;

  deltas_Wd = deltas_Wd / numImages;
  deltas_bd = deltas_bd / numImages;
end