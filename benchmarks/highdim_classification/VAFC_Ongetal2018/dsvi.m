function [F, mu, C] = dsvi(mu, C, loglik, logprior, options)
%function [F, mu, C] = dsvi(mu, C, loglik, logprior, options)
%
% What it does: applies doubly stochastic variational inference 
% in a joint probability model with some arbitrary, but 
% differentiable wrt parameters, log likelihood and log prior.
%
% Inputs 
%         - mu: D x 1 mean vector of the variational distribution.  
%         - C: scale matrix of the variational distribution.
%              If the size is D x D, then a full lower triangular positive 
%              definite (Cholesky) matrix is learned.  
%              If the size is D x 1, then a fully factorized approximation is
%              learned. 
%         - loglik: a structure containing a function handle and the input arguments for
%            the log likelihood. 
%         - logprior: a structure containing a function handle and the input arguments for
%            the log prior over the parameters.   
%         - options: 
%                options(1) is the number of stochastic approximation iterations.  
%                options(2) is the fixed learning rate for mu (while
%                            0.1*options(2) is the corresponding rate for C).
%                options(3) is the ratio between the full length of the dataset and the size of the minibatch
%                           (if training is done with a full dataset this is just 1) 
%                options(4) if 1, it uses as the standardized distribution the standard normal. 
%                           if 2, it uses a product of standard logistic distributions. 
%
% Outputs   
%         - F: a vector with all stochastic instantaneous values of the
%              lower bound. 
%         - mu: the final/learned value for mu. 
%         - C:  the final/learned value for C. 
%
% Examples: see demo_gprhypBodyfat.m, demo_gprhypBoston.m, demo_gprhypPendulum.m.    
%    
% Michalis Titsias (2014)

[D, D2] = size(C);
options(3) = 1;

whichStandDist = 1;  

if D2 == 1
    diagfull = 1; 
elseif D2 == D
    diagfull = 0;
    tmpC = triu(ones(D))';
else
    error('Something is wrong with the initial C: must be either D x D or D x 1.')
end

Niter = options(1); % Number of likelihood/gradient evaluations

F = zeros(1,Niter);


Edelta2_mu = zeros(length(mu),1);
Eg2_mu = zeros(length(mu),1);

Edelta2_C = zeros(length(C(:)),1);
Eg2_C = zeros(length(C(:)),1);

rho = 0.95;
eps_step = 10^-6;
oldEdelta2_mu = Edelta2_mu;
oldEg2_mu = Eg2_mu;
  
oldEdelta2_C = Edelta2_C;
oldEg2_C = Eg2_C;


for n = 1:Niter
%   
    
    if whichStandDist == 1      % Gaussian 
       z = randn(D,1);   
    elseif whichStandDist == 2  % Logistic distribution
       z = rand(D,1);
       z = log(z./(1-z));
    end
    
    if diagfull == 1
        theta = C.*z + mu;
    else
        theta = C*z + mu; 
    end
    
    q = size(loglik.inargs{1},2);
     
    theta_lambda = theta((q+1):(end-1));
    theta_g = theta(end);
    logprior.inargs{2}(2:q) = (exp(theta_g)^2)*exp(theta_lambda).^2;

    [g_lik dg_lik] = loglik.name(theta(1:q), loglik.inargs{:});
    [g_prior dg_prior] = logprior.name(theta, logprior.inargs{:},q);
    
    dg_lik = [dg_lik ; zeros(q,1)];
        
    g = g_lik + g_prior;
    dg = dg_lik + dg_prior;
    

     
    % Stochastic gradient wrt (mu,C) of the lower bound 
    if diagfull == 1 
       dmu = dg;
       dC = (dg.*z) + 1./C;
    else     
       dmu = dg;
       dC = (dg*z').*tmpC + diag(1./diag(C));
    end
    
    % Learning rates for mu
   
    Eg2_mu = rho*oldEg2_mu + (1-rho)*dmu.^2;
    Change_delta_mu = sqrt(oldEdelta2_mu + eps_step)./sqrt(Eg2_mu + eps_step).*dmu;

    mu = mu + Change_delta_mu;
    Edelta2_mu = rho*oldEdelta2_mu + (1- rho)*Change_delta_mu.^2; 
    
    oldEdelta2_mu = Edelta2_mu;
    oldEg2_mu = Eg2_mu;
    
    % Learning rates for C
    
    vec_dC = dC(:);
    Eg2_C = rho*oldEg2_C + (1-rho)*vec_dC.^2;
    Change_delta_C = sqrt(oldEdelta2_C + eps_step)./sqrt(Eg2_C + eps_step).*vec_dC;

    C = C + vec2mat(Change_delta_C,size(dC,1),size(dC,2));
    Edelta2_C = rho*oldEdelta2_C + (1- rho)*Change_delta_C.^2; 
    
    oldEdelta2_C = Edelta2_C;
    oldEg2_C = Eg2_C;
    
    % Update the variational parameters
%    mu = mu + ro*dmu; 
%    C = C + (0.1*ro)*dC;
    
    if diagfull == 1
       C(C<=1e-4)=1e-4;              % constraint (for numerical stability and positive definitenes)  
       logdetC = sum(log(C));
    else 
       keep = diag(C);
       keep(keep<=1e-4)=1e-4;        % constraint (for numerical stability and positive definitenes)
       C = C + (diag(keep - diag(C)));
       logdetC = sum(log(diag(C)));
    end
    
    % entropy of the standardized distribution
    if whichStandDist == 1
       entr = 0.5*D + 0.5*D*log(2*pi);  % Gaussian 
    elseif whichStandDist == 2         
       entr = D*2;                      % product of lostistics
    end
    
    % stochastic value of the lower bound 
    F(n) = g + logdetC + entr;
%    
    if ~(mod(n,10))
        n
        F(n)
        
        
    end
    
end
