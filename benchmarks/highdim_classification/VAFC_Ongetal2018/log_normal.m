function [g, dg] = lognormal(theta, m, K)
%


[D D2] = size(K);

if D == D2  % the covariance matrix is full
%     
   L = chol(K)';
   d = theta - m; 
   ld = L\d; 
   g = - 0.5*D*log(2*pi) - sum(log(diag(L)))  - 0.5*(ld'*ld); 
   dg = - (L'\ld);
%   
else % the covariance matrix is diagonal
% 
   d = theta - m;  
   g = - 0.5*D*log(2*pi) - 0.5*sum(log(K))  - 0.5*sum((d.^2)./K); 
   dg = - d./K;   
%
end


