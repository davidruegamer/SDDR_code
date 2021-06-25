function [L_mu,L_B,L_d,g] = gradient_compute(theta,B,z,d,eps,Ytr,Xtr,type,loglik,logprior,SiginvB,SiginvD,gradient_remove,q)


p = size(B,2);


if strcmp(type,'logistic_normal')
    
    [g_lik dg_lik] = loglik.name(theta, loglik.inargs{:});
    [g_prior dg_prior] = logprior.name(theta, logprior.inargs{:});
    
    g = g_lik + g_prior;
    delta_logh = dg_lik + dg_prior;
    
   
   
end
if strcmp(type,'logistic_GLMM_p1')        
    [g_lik dg_lik] = loglik.name(theta(1:(end-1)), loglik.inargs{:});
    [g_prior dg_prior] = logprior.name(theta, logprior.inargs{:}, b_n);
    
    dg_lik = [dg_lik ; 0];
    g = g_lik + g_prior;
    delta_logh = dg_lik + dg_prior;
end

if strcmp(type,'logistic_horseshoe')        
    [g_lik dg_lik] = loglik.name(theta(1:q), loglik.inargs{:});
    [g_prior dg_prior] = logprior.name(theta, logprior.inargs{:},q);
    
    dg_lik = [dg_lik ; zeros(q,1)];
    g = g_lik + g_prior;
    delta_logh = dg_lik + dg_prior;
end
%delta_h = exp(delta_h);
if (p > 0)
    Bz_deps = B*z + d.*eps;
else
    Bz_deps = d.*eps;
end
%Siginv = D^(-2) - D^(-2)*B/(eye(p) + B'*D^(-2)*B)*B'*D^(-2);

if(p > 0)
Dinv2B = bsxfun(@times,B,1./d.^2); %D^-2*B
end
DBz_deps = bsxfun(@times,Bz_deps,1./d.^2);  %D^-2 * Bz_deps



if gradient_remove == 0
    Half1 = DBz_deps;
    Half2 = Dinv2B/(eye(p) + B'*Dinv2B)*B'*DBz_deps;

    SBBSB = Half1*(Half1'*B) - Half1*(Half2'*B)-Half2*(Half1'*B)+Half2*(Half2'*B);
    SBBSD = Half1.*Half1.*d-Half1.*Half2.*d-Half2.*Half2.*d_Half2.*Half2.*d;
    
    L_mu = delta_logh  + (Half1-Half2);
    L_B = delta_logh*z'+SiginvB+(Half1-Half2)*z'-SBBSB;    
    L_d = delta_logh.*eps + SigInvD + (Half1 - Half2).*eps - SBSSD;
    
end
if gradient_remove == 1
    Half1 = DBz_deps;
    if (p > 0)
        Half2 = Dinv2B/(eye(p) + B'*Dinv2B)*B'*DBz_deps;
        L_mu = delta_logh + (Half1-Half2);
        L_B = delta_logh*z'+(Half1-Half2)*z';
        L_d = delta_logh.*eps + (Half1 - Half2).*eps;
    else
        L_mu = delta_logh;
        L_B = delta_logh*z';
        L_d = delta_logh.*eps + Half1.*eps;
    end

end
