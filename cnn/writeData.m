function y = writeData(iteration, cost)
  y = zeros(1)
  numImages = size(probs, 2);

  for i = 1:numImages
    y(labels(i), i) = 1;
  end
end