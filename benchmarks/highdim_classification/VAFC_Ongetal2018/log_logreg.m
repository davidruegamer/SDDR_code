function [g, dg] = log_logreg(theta, X, Y, T)
%function [g, dg] = log_logreg(theta, X, Y, T)
%
% 

F = X*theta; 

YF = -Y.*F;

m = max(0,YF);

g = - sum( m + log( exp(-m) + exp( YF - m )) ); 

S = sigmoid(F);

dg = X'*(T - S);



