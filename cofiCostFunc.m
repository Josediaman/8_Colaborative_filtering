function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
% J: Cost of the regresion with theta.
% grad: Gradient of J.
% params: Parameters of the regresion.
% Y: Training examples (valorations).
% R: Positions of valorations.
% num_users: number of users.
% num_movies: Number of movies.
% num_features: Number of features.
% lambda: Parameter of the regularization.




X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));


predictions=X*Theta';
Errors=(predictions-Y);
J=(1/2)*sum(sum(R.*(Errors.^2)))+(lambda/2)*(sum(sum(Theta.^2)))+(lambda/2)*(sum(sum(X.^2)));


X_grad=(R.*Errors)*Theta+lambda*X;
Theta_grad=(R.*Errors)'*X+lambda*Theta;


grad = [X_grad(:); Theta_grad(:)];


end





