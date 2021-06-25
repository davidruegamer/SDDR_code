function [LowerB,B,mu,d,ADA,SiginvB,SiginvD,L_mu,L_B,L_d,logprior] = F_log_horseshoe(B, mu, d, Xtr, Ytr,type,loglik,logprior,SiginvB,SiginvD,ADA,p,gradient_remove)

%%%%% This script uses logistic likelihood with normal prior %%%%%%

rho = ADA.rho;
eps_step = ADA.eps_step;
oldEdelta2_mu = ADA.Edelta2_mu;
oldEg2_mu = ADA.Eg2_mu;
  
oldEdelta2_B = ADA.Edelta2_B;
oldEg2_B = ADA.Eg2_B;

oldEdelta2_d = ADA.Edelta2_d;
oldEg2_d = ADA.Eg2_d;

zeps = randn(length(mu)+p,1)';
z = zeps(1:p)';
eps = zeps((p+1):end)';

theta = mu + B*z + d.*eps;

q = size(Xtr,2);
theta_lambda = theta((q+1):(end-1));
theta_g = theta(end);
logprior.inargs{2}(2:q) = (exp(theta_g)^2)*exp(theta_lambda).^2;

[L_mu,L_B,L_d,g] = gradient_compute(theta,B,z,d,eps,Ytr,Xtr,type,loglik,logprior,SiginvB,SiginvD,gradient_remove,q);

L_B(~tril(ones(size(L_B))))  = 0;


%%%%%%%%%%%%%%%
%%% The update below can be condensed into one straightforward update
%%% I decided to split it into three parts for testing purpose of specific
%%% parameters
%%% Please condense the codes back when everything is finalized 
%%%%%%%%%%%%%%%
%% mu update
   
    ADA.Eg2_mu = rho*oldEg2_mu + (1-rho)*L_mu.^2;
    Change_delta_mu = sqrt(oldEdelta2_mu + eps_step)./sqrt(ADA.Eg2_mu + eps_step).*L_mu;
    
    
    mu = mu + Change_delta_mu;
    ADA.Edelta2_mu = rho*oldEdelta2_mu + (1- rho)*Change_delta_mu.^2; 
    
  %  oldEdelta2_mu = Edelta2_mu;
  %  oldEg2_mu = Eg2_mu;

%% B update

if (p > 0)
    vecL_B = L_B(:);
     
    ADA.Eg2_B = rho*oldEg2_B + (1-rho)*vecL_B.^2;
    Change_delta_B = sqrt(oldEdelta2_B + eps_step)./sqrt(ADA.Eg2_B + eps_step).*vecL_B;
    
    
    B = B + vec2mat(Change_delta_B,size(B,1),size(B,2));
    ADA.Edelta2_B = rho*oldEdelta2_B + (1- rho)*Change_delta_B.^2;  
end
  %  oldEdelta2_B = Edelta2_B;
   % oldEg2_B = Eg2_B;
 
%% d update

    ADA.Eg2_d = rho*oldEg2_d + (1-rho)*L_d.^2;
    Change_delta_d = sqrt(oldEdelta2_d + eps_step)./sqrt(ADA.Eg2_d + eps_step).*L_d;

    d = d + Change_delta_d;
    ADA.Edelta2_d = rho*oldEdelta2_d + (1- rho)*Change_delta_d.^2; 
    
  %  oldEdelta2_d = Edelta2_d;
  %  oldEg2_d = Eg2_d;

  %  theta = mu + B*z + d.*eps;
 
  if (p > 0)
    loghtheta = g;
    Bz_deps = B*z + d.*eps;
    DBz_deps = bsxfun(@times,Bz_deps,1./d.^2); 
    
    Dinv2B = bsxfun(@times,B,1./d.^2);
    %Siginv = diag(1./d.^2) - Dinv2B/(eye(p) + B'*Dinv2B)*Dinv2B';

    Half1 = DBz_deps;
    Half2 = Dinv2B/(eye(p) + B'*Dinv2B)*B'*DBz_deps;
    
    Blogdet = logdet(eye(p) + bsxfun(@times,B, 1./(d.^2))'*B) + sum(log((d.^2)));
    LowerB = loghtheta + q/2*log(2*pi) + 1/2*Blogdet + 1/2*Bz_deps'*(Half1-Half2);
    
  else
    loghtheta = g;
    Bz_deps = d.*eps;

    Blogdet = logdet(eye(p)) + sum(log((d.^2)));
    LowerB = loghtheta + q/2*log(2*pi) + 1/2*Blogdet + 1/2*sum((Bz_deps.*(1./d.^2).*Bz_deps));     
  end
  