


%% ................................................
%% ................................................
%%  COLABORATIVE FILTERING
%% ................................................
%% ................................................





%% 1. Clear and Close Figures
clear ; close all; clc





%% ========= Part 1: Data ================
fprintf('\n \nDATA\n.... \n \n \n');   





%% 2. Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Add your own file

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


fprintf('Loading data ...\n'); 
%%%%%%********Select archive********   
load('ex8_movies.mat'); 
fprintf('(Y) (10 items)\n\n');   
[Y(1:10,1:10)]
fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% 3. Plotting Data
fprintf('\n\nPlot  Y\n\n');   
imagesc(Y);
ylabel('Movies');
xlabel('Users');
fprintf('Program paused. Press enter to continue. \n \n \n');
pause;





%% ======= Part 2: Entering ratings for a new user ==========
fprintf('RATINGS OF A NEW USER\n.....................\n \n \n');





%% 4. Initial values of the new customer.
movieList = loadMovieList();
my_ratings = zeros(1682, 1);
% Ratings of the new customer:
%%%%%%********Select ratings of new customer********   
my_ratings(1) = 4;
my_ratings(98) = 2;
my_ratings(7) = 3;
my_ratings(12)= 5;
my_ratings(54) = 4;
my_ratings(64)= 5;
my_ratings(66)= 3;
my_ratings(69) = 5;
my_ratings(183) = 4;
my_ratings(226) = 5;
my_ratings(355)= 5;
fprintf('\n\nNew user ratings:\n');
for i = 1:length(my_ratings)
    if my_ratings(i) > 0 
        fprintf('Rated %d for %s\n', my_ratings(i), ...
                 movieList{i});
    end
end

% Add ratings of new customer to training examples
Y = [my_ratings Y];
R = [(my_ratings ~= 0) R];
fprintf('\nProgram paused. Press enter to continue.\n\n\n\n');
pause;





%% ======= Part 3: Learning Movie Ratings ============
fprintf('TRAINING COLLABORATIVE FILTERING\n................................\n \n \n \n');





%  Normalize Ratings
[Ynorm, Ymean] = normalizeRatings(Y, R);
num_users = size(Y, 2);
num_movies = size(Y, 1);
%%%%%%********Select number of features********   
num_features = 10;


% Set Initial Parameters (Theta, X)
X = randn(num_movies, num_features);
Theta = randn(num_users, num_features);
initial_parameters = [X(:); Theta(:)];


% Set Regularization
%%%%%%********Select lambda and iterations********   
lambda = 10;
max_iter=100;
options = optimset('GradObj', 'on', 'MaxIter', max_iter);
theta = fmincg (@(t)(cofiCostFunc(t, Ynorm, R, num_users, num_movies, ...
                                num_features, lambda)), ...
                initial_parameters, options);
X = reshape(theta(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(theta(num_movies*num_features+1:end), ...
                num_users, num_features);

fprintf('\nProgram paused. Press enter to continue.\n\n\n\n');
pause;





%% ======== Part 4: Recommendation for the new customer ========
fprintf('RECOMMENDATION FOR NEW CUSTOMER\n...............................\n \n \n ');





% Predictions
p = X * Theta';


% Predictions of the new customer
my_predictions = p(:,1) + Ymean;
movieList = loadMovieList();


% Recommendations for the new customer
[r, ix] = sort(my_predictions, 'descend');
fprintf('\nTop recommendations for new customer:\n');
for i=1:10
    j = ix(i);
    fprintf('Predicting rating %.1f for movie %s\n', my_predictions(j), ...
            movieList{j});
end

