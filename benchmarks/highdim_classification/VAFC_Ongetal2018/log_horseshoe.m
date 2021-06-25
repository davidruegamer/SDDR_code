function [g, dg] = log_horseshoe(theta, m, K, q)
% m is the mean
% K is the covariance
% We consider only diagonal covariance here

theta_beta = theta(2:q);
theta_nu = theta((q+1):(end-1));

g_beta0 = -0.5*log(2*pi*K(1)) - 1/(2*K(1))*theta(1)^2;
g_beta = -0.5*log(2*pi) - (theta_nu + theta(end)) - (theta_beta.^2)./(2*exp(2*(theta_nu + theta(end))));
g_nu = log(2/pi) + theta((q+1):end) - log(1 + exp(2*theta((q+1):end)));

g = sum([g_beta0 ; g_beta ; g_nu]);

dg_beta0 = - theta(1)/K(1);
dg_beta = -theta(2:q)./exp(2*(theta_nu + theta(end)));
dg_nu = - 2*exp(2*theta_nu)./(1 + exp(2*theta_nu)) + (theta_beta.^2)./(exp(2*(theta_nu + theta(end)))); 
dg_nu_qone = 1 - length(dg_nu) - 2*exp(2*theta(end))./(1 + exp(2*theta(end))) + sum((theta_beta.^2)./(exp(2*(theta_nu + theta(end))))); 

dg = [dg_beta0 ; dg_beta  ; dg_nu ; dg_nu_qone];
end


