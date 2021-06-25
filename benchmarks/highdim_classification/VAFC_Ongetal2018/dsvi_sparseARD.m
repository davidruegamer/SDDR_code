function [F, mu, C] = dsvi_sparseARD(mu, C, loglik, options)
%function [F, mu, C] = dsvi_sparseARD(mu, C, loglik, options)
% dsvi_sparseARD(mu, C, loglik, options)
% What it does: it has a very similar form with the general dsvi function. 
%               The only difference is that the prior over parameters 
%               is an ARD Gaussian prior and the hyperpaparameters of this prior 
%               are removed analytically from the objective function.
%
% Inputs 
%         - mu: D x 1 mean vector of the variational distribution.  
%         - C:  D x 1 scale vector associated with a fully factorized approximation. 
%         - loglik: a structure containing a function handle and the input arguments for
%            the log likelihood. 
%         - options: 
%                options(1) is the number of stochastic approximation iterations.  
%                options(2) is the fixed learning rate for mu (while
%                            0.1*options(2) is the corresponding rate for C).
%                options(3) is the ratio between the full length of the dataset and the size of the minibatch
%                           (if training is done with a full dataset this is just 1) 
%
% Outputs   
%         - F: a vector with all stochastic instantaneous values of the
%              lower bound. 
%         - mu: the final/learned value for mu. 
%         - C:  the final/learned value for C. 
%
% Examples: see demo_logregColon, demo_logregLeukemia, demo_logregDukeBreastCancer.
%    
% Michalis Titsias (2014)


D = length(mu);

[D, D2] = size(C);


if options(3) == 0
    options(3) = 1;
end

% Ratio between the full length of the dataset and the minibatch
% This simple will be eqaul to 1 if all the data are used 
Nn = options(3); 

Niter = options(1); % Number of likelihood/gradient evaluations
ro = options(2) ;   % Learning rate

F = zeros(1,Niter);

%%%% ADDED  
if D2 == 1
    diagfull = 1; 
elseif D2 == D
    diagfull = 0;
    tmpC = triu(ones(D))';
else
    error('Something is wrong with the initial C: must be either D x D or D x 1.')
end
%%%%%%%%

for n = 1:Niter

    z = randn(D,1);   
%%%% ADDED    
    if diagfull == 1
        theta = C.*z + mu;
    else
        theta = C*z + mu; 
    end
%%%%%%%%
    %theta = C.*z + mu;
    
    [g_lik dg_lik] = loglik.name(theta, loglik.inargs{:});
    
    dg = Nn*dg_lik;
    
    C2 = C.*C;
    Cmu = C2 + mu.^2;
    
    % Stochastic gradient update of the parameters
    
    if diagfull == 1 
        dmu = dg - mu./Cmu;
        dC = (dg.*z) + 1./C - C./Cmu;
    else     
       dmu = dg - mu./Cmu;
       dC = (dg*z').*tmpC + diag(1./diag(C)) - C./Cmu;
    end
    

    
    
    mu = mu + ro*dmu;      
    C = C + (0.1*ro)*dC;
    
    if diagfull == 1
       C(C<=1e-4)=1e-4;              % constraint (for numerical stability and positive definitenes)  
       logdetC = sum(log(C));
    else 
       keep = diag(C);
       keep(keep<=1e-4)=1e-4;        % constraint (for numerical stability and positive definitenes)
       C = C + (diag(keep - diag(C)));
       logdetC = sum(log(diag(C)));
    end

    C(C<=1e-4)=1e-4;        % constraint (for numerical stability)  
       
    % stochastic value of the lower bound
    % (data term plus the optimal KL term)
    F(n) = Nn*g_lik + 0.5*sum(log(C2./Cmu)); 
%    
end
 
