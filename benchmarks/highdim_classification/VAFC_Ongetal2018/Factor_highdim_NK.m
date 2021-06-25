% These codes uses the setup of the dataset in the paper written by 

% Michalis K. Titsias and Miguel Lazaro-Gredilla. 
% Doubly Stochastic Variational Bayes for non-Conjugate Inference.
% Proceedings of the 31st International Conference on Machine
% Learning (ICML), Beijing, China, 2014. JMLR: W&CP volume 32.

% randn('seed',1);
% rand('seed',1);

rng(1);
%parpool(20);

for p = [4, 20]
for data_pick = 3
clearvars -except data_pick p;  
  
switch data_pick 
    case 1
        disp('Running Colon data...')
        load('colon_train_test.mat','Ytr','Yts','Xtr','Xts');
        [Ntr, D] = size(Xtr);
        [Nts, D] = size(Xts);
    case 2
        disp('Running Leukemia data...')
        load('leukemia_train_test.mat','Ytr','Yts','Xtr','Xts');
        [Ntr, D] = size(Xtr);
        [Nts, D] = size(Xts);
    case 3
        load('duke_train_test.mat','Ytr','Yts','Xtr','Xts');
        [Ntr, D] = size(Xtr);
        [Nts, D] = size(Xts);
        Ytr=Ytr';
        Yts=Yts';
        
    otherwise
        disp('Please pick one of the three datasets')
end


q = size(Xtr,2);


%% Initializing Normal-dist logistic 


% log likelihood function
loglik.name = @log_logreg;     % logistic regression log likelihood
loglik.inargs{1} = Xtr;        % input data 
loglik.inargs{2} = Ytr;        % binary outputs encoded as -1,1
loglik.inargs{3} = (Ytr+1)/2;  % binary outputs encoded as 0,1 (just for convenience)

% log prior
logprior.name = @log_horseshoe;  % prior over the hyperparameters
logprior.inargs{1} = zeros(2*q,1);   % mean of the prior
logprior.inargs{2} = zeros(2*q,1) + 10; % variances of the prior

gradient_remove = 1;
F_mus=cell(20,1);
T_all=cell(20,1);
F_LBs=F_mus;
%% Initializing variational parameters starting point
parfor seedi=1:20
fprintf('seed=\n',[seedi]);
rng(seedi);
mu0 = zeros(2*(q),1); % beta0+betaj+lambdaj+g
mu0(end) = 5; % g = 5
d0 = ones(2*q,1)*1;

B0 = zeros(2*q,p) + 0.001;
B0(logical(eye(size(B0)))) = 0.001;
B0(find(triu(B0,1))) = 0;

B = B0;
mu = mu0;
d = d0;

type = 'logistic_horseshoe'

iter = 1;
niter = 5000;
LowerB = zeros(niter,1);

Edelta2_mu = zeros(length(mu),1);
Eg2_mu = zeros(length(mu),1);

Edelta2_B = zeros(length(B(:)),1);
Eg2_B = zeros(length(B(:)),1);

Edelta2_d = zeros(length(d),1);
Eg2_d = zeros(length(d),1);


ADA.rho = 0.95;
ADA.eps_step = 10^-6;
ADA.Edelta2_mu = Edelta2_mu;
ADA.Eg2_mu = Eg2_mu;

ADA.Edelta2_B = Edelta2_B;
ADA.Eg2_B = Eg2_B;

ADA.Edelta2_d = Edelta2_d;
ADA.Eg2_d = Eg2_d;

Dinv2B = bsxfun(@times,B,1./d.^2);
Dinv2Bterm = Dinv2B/(eye(p)+B'*Dinv2B);
SiginvB = Dinv2B-Dinv2Bterm*(Dinv2B'*B);
SiginvD= 1./d-sum(Dinv2Bterm.*bsxfun(@times,B,1./d),2);

%Siginv = diag(1./d.^2) - Dinv2B/(eye(p) + B'*Dinv2B)*Dinv2B';

StoreLB = zeros(niter,1);


Storemu = zeros(niter,length(mu));
Storesig =  zeros(niter,length(mu));

TStart=tic;
for iter = 1:niter
    [LowerB,B,mu,d,ADA,SiginvB,SiginvD,L_mu,L_B,L_d] = F_log_horseshoe(B, mu, d, Xtr, Ytr,type,loglik,logprior,SiginvB,SiginvD,ADA,p,gradient_remove);
    StoreLB(iter) = LowerB;
    Storemu(iter,:) = mu;
    Storesig(iter,:) = sum(B.^2,2) + d.^2;
    if ~(mod(iter,100))
        iter
        StoreLB(iter)
%         F_mu = mu;
%         F_d = d;
%         F_B = B;
%         F_LB = StoreLB;
    end
%     switch data_pick 
%         case 1
%             disp('Saving Colon data...')
%             save('Output/F_Colon_',num2str(seedi),'.mat'),'F_LB','TElapsed','F_mu','Xtr','Ytr','Xts','Yts','iter')
%         case 2
%             disp('Saving Leukemia data')
%             save('Output/F_leuk_school_',num2str(seedi),'.mat'),'F_LB','TElapsed','F_mu','Xtr','Ytr','Xts','Yts','iter')
%         case 3
%             disp('Saving Duke Cancer data')
%             save('Output/F_duke_',num2str(seedi),'.mat'),'F_LB','TElapsed','F_mu','Xtr','Ytr','Xts','Yts','iter')
%     end
%    end    
    
end
TElapsed=toc(TStart);
F_mus{seedi,1}=mu;
F_LBs{seedi,1}=StoreLB;
T_all{seedi,1}=TElapsed;
% F_mu = mu;
% F_d = d;
% F_B = B;
% F_LB = StoreLB;
end;

for seedi=1:20
F_LB=F_LBs{seedi,1};
F_mu=F_mus{seedi,1};
TElapsed=T_all{seedi,1};

if (p == 4)
switch data_pick 
    case 1
        disp('Saving Colon data...')
            save(strcat('Output_NK/F_Colon_p4_',num2str(seedi),'.mat'),'F_LB','TElapsed','F_mu','Xtr','Ytr','Xts','Yts')
    case 2
        disp('Saving Leukemia data')
            save(strcat('Output_NK/F_leuk_p4_',num2str(seedi),'.mat'),'F_LB','TElapsed','F_mu','Xtr','Ytr','Xts','Yts')
    case 3
        disp('Saving Duke Cancer data')
            save(strcat('Output_NK/F_duke_p4_',num2str(seedi),'.mat'),'F_LB','TElapsed','F_mu','Xtr','Ytr','Xts','Yts')
end
end
if (p == 20)
switch data_pick 
    case 1
        disp('Saving Colon data...')
            save(strcat('Output_NK/F_Colon_p20_',num2str(seedi),'.mat'),'F_LB','TElapsed','F_mu','Xtr','Ytr','Xts','Yts')
    case 2
        disp('Saving Leukemia data')
            save(strcat('Output_NK/F_leuk_p20_',num2str(seedi),'.mat'),'F_LB','TElapsed','F_mu','Xtr','Ytr','Xts','Yts')
    case 3
        disp('Saving Duke Cancer data')
            save(strcat('Output_NK/F_duke_p20_',num2str(seedi),'.mat'),'F_LB','TElapsed','F_mu','Xtr','Ytr','Xts','Yts')
end
end

end
end
end