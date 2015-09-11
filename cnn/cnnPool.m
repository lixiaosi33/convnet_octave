function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%     

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);
idim = convolvedDim / poolDim;

if(isinteger(idim))
  error('convolvedim not divisible by poolDim')
endif

pooledFeatures = zeros(idim, idim, numFilters, numImages);

% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures x numImages 
%   matrix pooledFeatures, such that
%   pooledFeatures(poolRow, poolCol, featureNum, imageNum) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region. 
%   
%   Use mean pooling here.

%%% YOUR CODE HERE %%%
for imageNum = 1:numImages
  for filterNum = 1:numFilters
    filter = convolvedFeatures(:, :, filterNum, imageNum);

    for x = 1:idim
      for y = 1:idim
      	mat = filter(x*poolDim - poolDim +1 : x*poolDim , y*poolDim - poolDim + 1 : y*poolDim);
      	pooledFeatures(x, y, filterNum, imageNum) = mean(mat(:));
      end
    end

  end
end

end

