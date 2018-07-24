function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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

% Use costFunction to get original J and grad values
[J, grad] = costFunction(theta, X, y);

% Compute J regularization term, add to J
% Do not need to augment J_reg because already discounted theta(1) and J=[1x1]
J_reg = lambda * sum(theta(2:size(theta)) .^ 2) / (2 * m);
J += J_reg;

% Compute grad regularization term, augment with 0 for j=0, add to grad
grad_reg = lambda * [0; theta(2:size(theta))] / m;
grad += grad_reg;

% =============================================================

end
