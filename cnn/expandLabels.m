function y = expandLabels(labels, probs)
  y = zeros(size(probs));
  numImages = size(probs, 2);

  for i = 1:numImages
    y(labels(i), i) = 1;
  end
end