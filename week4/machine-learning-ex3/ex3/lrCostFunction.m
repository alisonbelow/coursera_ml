function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

% Copy/pasted following code from week3/ex2 assignment
h_theta = sigmoid(X*theta);     

% Need to flip y[100x1] to y'[1x100] for multiplication with h_theta[100x1]
J = ( -1 * y' * log(h_theta) - (1 - y)' * log(1 - h_theta) ) / m;

% Need to flip (h_theta-y) from [100x1] to [1x100] for mult with X[100x3]
% Resulting product is [1x3], flip once more to be [3x1]
grad = ((h_theta - y)' * X) / m;
grad = grad';

% Compute J regularization tderm, add to J
% Do not need to augment J_reg because already discounted theta(1) and J=[1x1]
J_reg = lambda .* sum(theta(2:size(theta)) .^ 2) / (2 * m);
J += J_reg;

% Compute grad regularization term, augment with 0 for j=0, add to grad
grad_reg = lambda .* [0; theta(2:size(theta))] / m;
grad += grad_reg;




% =============================================================

grad = grad(:);

end
