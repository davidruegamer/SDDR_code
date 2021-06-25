############# Load libraries and helper scripts ################
rm(list=ls())
library(deepregression)
nrCores <- 1
load_all(path)
library(gamboostLSS)
library(gamlss)
library(bamlss)
source("formula_generator.R")
source("helper_functions.R")
source("dgp.R")
source("setup_model_comparison.R")

################# Define simulation settings ###################

settings <- expand.grid(n = c(300, 2500),
                        p1 = c(10, 75),
                        p2 = c(2),
                        pnl1 = c(10),
                        pnl2 = c(2), 
                        overlapl = c(2), 
                        overlapnl = c(0),
                        family = c("normal", 
                                   #"lognormal",
                                   #"negbin",
                                   "gamma", 
                                   "logistic")
)

nrsim <- 20

for(i in 1:nrow(settings)){
  
  cat("Working on setting", i, "...\n")
  
  set <- settings[i,]

  this_family <- as.character(set$family)
  outcome_trafo <- switch(this_family,
                          normal = function(n, pred1, pred2) 
                            rnorm(n = n, 
                                  mean = pred1,
                                  sd = exp(pred2)),
                          lognormal = function(n, pred1, pred2)
                            rLOGNO(n = n, mu = pred1, sigma = exp(pred2)),
                          negbin = function(n, pred1, pred2)
                            rNBI(n = n, 
                                 mu = exp(pred1), sigma = exp(pred2)),
                          beta = function(n, pred1, pred2)
                            rBE(n = n, 
                                mu = plogis(pred1), 
                                sigma = plogis(pred2)),
                          gamma = function(n, pred1, pred2)
                            rGA(n = n, 
                                mu = exp(pred1), 
                                sigma = exp(pred2)),
                          logistic = function(n, pred1, pred2)
                            rLO(n = n, mu = pred1, sigma = exp(pred2)))
  
  
  mclapply(1:nrsim, function(j){
    
    set.seed(j)
    
    data <- dgp(n = set$n,
                p1 = set$p1,
                p2 = set$p2,
                pnl1 = set$pnl1,
                pnl2 = set$pnl2,
                overlapl = set$overlapl,
                overlapnl = set$overlapnl,
                bias1 = -1,
                bias2 = -1,
                outcome_trafo = outcome_trafo,
                divideBy = 1+9*(this_family%in%c("negbin","gamma")))
    
    if(this_family=="beta"){
      data$y[data$y==1] <- 1-10e-9
      data$y[data$y==0] <- 10e-9
    }
    this_data <- cbind(y = data$y, data$X)
    
    all_fits <- try(fit_all_models(formula_list = data$list_of_formulae,
                                   data = this_data[data$train,],
                                   newdata = this_data[data$test,],
                                   distribution = this_family,
                                   iterations = ifelse(set$n==300, 
                                                       600, 
                                                       6000),# / (set$family=="gamma")*2,
                                   use_cv_dr=FALSE)
    )
    if(class(all_fits)=="try-error") return(NULL)
    
    results <- calculate_results(all_fits, data)
    results <- cbind(results, set)
    results$simnr = j
    
    saveRDS(results, file = paste0("results/model_comparisons/setting_",
                                   i, "_iteration_", j, ".RDS"))
    
  }, mc.cores = nrCores)

}
