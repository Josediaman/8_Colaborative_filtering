function [Ynorm, Ymean] = normalizeRatings(Y, R)
% Ynorm: norm of the training examples.
% Ymean: mean of the training examples (by colums).
% Y: Training examples (valorations).
% R: Positions of valorations.

[m, n] = size(Y);
Ymean = zeros(m, 1);
Ynorm = zeros(size(Y));
for i = 1:m
    idx = find(R(i, :) == 1);
    Ymean(i) = mean(Y(i, idx));
    Ynorm(i, idx) = Y(i, idx) - Ymean(i);
end

end
