function cost = softmaxCost(labels, probs)
%cnnConvolve Returns the convolution of the features given by W and b with
%the given images
%
% Parameters:
%  probs
%  labels
%
% Returns:
%  cost
% 
y = expandLabels(labels, probs);

ln_one_minus_a = log(1-probs);
ln_a =log(probs);
one_minus_y = 1 - y;

cost = -sum(mean(y.*ln_a + one_minus_y.*ln_one_minus_a, 2));
end