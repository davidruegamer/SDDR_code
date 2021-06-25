% This MATLAB software implements the doubly stochastic variational 
% inference (DSVI) method described in the following paper: 
% 
% Michalis K. Titsias and Miguel Lazaro-Gredilla. 
% Doubly Stochastic Variational Bayes for non-Conjugate Inference.
% Proceedings of the 31st International Conference on Machine
% Learning (ICML), Beijing, China, 2014. JMLR: W&CP volume 32.
% 
% The main routines are the functions dsvi.m (standard DSVI) 
% and dsvi_sparseARD.m (DSVI for automatic variable selection). 
% For further details see the documentation in the former functions 
% as well as the provided demos.  
% 
% The software of Rasmussen and  Williams (2006) that is used in the 
% GP hyperparameter learning experiments has been included for 
% convenience (directory toolbox_gpml). 
% 
% You can freely use this code for academic or research purposes 
% as long as you refer to the above publication.  
% 
% Copyright (c) Michalis Titsias and Miguel Lazaro-Gredilla (2014)
   

for data_pick = 1:3

clearvars -except data_pick
randn('seed',1);
rand('seed',1);

tic
switch data_pick 
    case 1
        disp('Running Colon data...')
        load('colon-cancer.mat')

        [n, D] = size(X);
        X = [ones(n,1), full(X)];  

        Class1 = find(Y==1); 
        Class2 = find(Y==-1); 
        N1 = length(Class1);
        N2 = length(Class2); 
        N1tr = 12; 
        N2tr = 30;

        ch1 = randperm(N1); 
        ch2 = randperm(N2); 
        ch = [ch1(1:N1tr), ch2(1:N2tr)];
               
        Xts = [X(ch1(N1tr+1:end),:); X(ch2(N2tr+1:end),:)];
        
            
        Ytr = [Y(ch1(1:N1tr)); Y(ch2(1:N2tr))];
        Xtr = [X(ch1(1:N1tr),:); X(ch2(1:N2tr),:)];   
        
        %%% Standardizating them hoping for good convergence
        Xts(:,2:end) = (Xts(:,2:end) - repmat(min(Xtr(:,2:end)),size(Xts,1),1))./(repmat(max(Xtr(:,2:end)),size(Xts,1),1) - repmat(min(Xtr(:,2:end)),size(Xts,1),1));
        Xtr(:,2:end) = (Xtr(:,2:end) - repmat(min(Xtr(:,2:end)),size(Xtr,1),1))./(repmat(max(Xtr(:,2:end)),size(Xtr,1),1) - repmat(min(Xtr(:,2:end)),size(Xtr,1),1));
        
        Ntr = size(Xtr,1); 

        Yts = [Y(ch1(N1tr+1:end)); Y(ch2(N2tr+1:end))];
        

        
        Nts = size(Xts,1);
    case 2
        disp('Running Leukemia data...')
        load leukemia.mat;

        [Ntr, D] = size(Xtr);
        [Nts, D] = size(Xts);

        Xtr = [ones(Ntr,1), full(Xtr)];  
        Xts = [ones(Nts,1), full(Xts)];
        
        %%% Standardizating them hoping for good convergence
        Xts(:,2:end) = (Xts(:,2:end) - repmat(min(Xtr(:,2:end)),size(Xts,1),1))./(repmat(max(Xtr(:,2:end)),size(Xts,1),1) - repmat(min(Xtr(:,2:end)),size(Xts,1),1));
        Xtr(:,2:end) = (Xtr(:,2:end) - repmat(min(Xtr(:,2:end)),size(Xtr,1),1))./(repmat(max(Xtr(:,2:end)),size(Xtr,1),1) - repmat(min(Xtr(:,2:end)),size(Xtr,1),1));
        
    case 3
        load duke.mat;

        [Ntr, D] = size(Xtr);
        [Nts, D] = size(Xts);

        % Add an extra column in X with ones for the bias term in 
        % logistic regression 
        Xtr = [ones(Ntr,1), full(Xtr)];  
        Xts = [ones(Nts,1), full(Xts)];
        
        %%% Standardizating them hoping for good convergence
        Xts(:,2:end) = (Xts(:,2:end) - repmat(min(Xtr(:,2:end)),size(Xts,1),1))./(repmat(max(Xtr(:,2:end)),size(Xts,1),1) - repmat(min(Xtr(:,2:end)),size(Xts,1),1));
        Xtr(:,2:end) = (Xtr(:,2:end) - repmat(min(Xtr(:,2:end)),size(Xtr,1),1))./(repmat(max(Xtr(:,2:end)),size(Xtr,1),1) - repmat(min(Xtr(:,2:end)),size(Xtr,1),1));
        
        
    otherwise
        disp('Please pick one of the three datasets')
end




% log likelihood function
loglik.name = @log_logreg;     % logistic regression log likelihood
loglik.inargs{1} = Xtr;        % input data 
loglik.inargs{2} = Ytr;        % binary outputs encoded as -1,1
loglik.inargs{3} = (Ytr+1)/2;  % binary outputs encoded as 0,1 (just for convenience)

% log prior
logprior.name = @log_horseshoe;  % prior over the hyperparameters
logprior.inargs{1} = zeros(2*(D+1),1);   % mean of the prior
logprior.inargs{2} = zeros(2*(D+1),1) + 10; % variances of the prior

dec = 0.95; 

options = zeros(1,10); 
options(1) = 10000;    % number of iterations per stage
options(2) = 0.05/size(Xtr,1);   % initial value of the learning rate

options(4) = 1;   % (1 - Gaussian standard distribution)  (1 - Gaussian standard distribution)

mu = zeros(2*(D+1),1);
mu(end) = 5;
C = diag(1*ones(2*(D+1),1));

iters = 1;  % number of optimization stages (each stage takes options(1) iterations)

ops = options; 
tic;

[F, mu, C] = dsvi(mu, C, loglik, logprior, options);

DS_mu = mu;
DS_C = C;
DS_LB = F;


switch data_pick 
    case 1
        disp('Saving Colon data...')
        save('Output/DS_Colon.mat','DS_mu','DS_LB','Xtr','Ytr','Xts','Yts')
    case 2
        disp('Saving Leukemia data')
        save('Output/DS_leuk.mat','DS_mu','DS_LB','Xtr','Ytr','Xts','Yts')
    case 3
        disp('Saving Duke Cancer data')
        save('Output/DS_duke.mat','DS_mu','DS_LB','Xtr','Ytr','Xts','Yts')
end

end
