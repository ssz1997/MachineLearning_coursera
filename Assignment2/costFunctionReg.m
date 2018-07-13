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


z = X * theta;
h_z = sigmoid(z);
for i = 1:m
  J += -y(i)*log(h_z(i)) - (1-y(i))*log(1-h_z(i));
endfor
J = J/m;
for i = 2:length(theta)
  J += lambda / 2 / m * theta(i) * theta(i)

grad(1) = sum((h_z-y) .* X(:,1))/m;

for i = 2:length(theta)
  grad(i) = sum((h_z-y) .* X(:,i))/m + lambda / m * theta(i);
endfor



% =============================================================

end
