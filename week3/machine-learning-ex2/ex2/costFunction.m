function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
% Note: grad should have the same dimensions as theta
%

h_theta = sigmoid(X*theta);     % Compute h_theta 

% Need to flip y[100x1] to y'[1x100] for multiplication with h_theta[100x1]
J = ( -1 * y' * log(h_theta) - (1 - y)' * log(1 - h_theta) ) / m;

% grad dim must = theta[3x1]
% Need to flip (h_theta-y) from [100x1] to [1x100] for mult with X[100x3]
% Resulting product is [1x3], flip once more to be [3x1]
grad = ((h_theta - y)' * X) / m;
grad = grad';




% =============================================================

end
