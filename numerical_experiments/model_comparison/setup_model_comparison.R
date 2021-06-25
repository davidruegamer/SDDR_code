###############################################################

# families across different packages
dist <- matrix(c(
  "Family",   "deepregression", "bamlss",           "gamlss",    "mboostLSS",
  ##################################################################################
  "normal",   "normal",         "gaussian_bamlss",  "NO",        "GaussianLSS",
  "lognormal","log_normal",     "lognormal_bamlss", "LOGNO",     "as.families('LOGNO')",
  "negbin",   "negbinom",       "nbinom_bamlss",    "NBII",      "NBinomialLSS",
  "beta",     "betar",          "beta_bamlss",      "BE",        "BetaLSS",
  "gamma",    "gammar",         "gamma_bamlss",     "GA",        "GammaLSS",
  "logistic", "logistic",       "glogis_bamlss",     "LO",        "as.families('LO')"#,
#  "ZIP",      "zip",            "zip_bamlss",       "ZIP",       "ZIPoLSS",
#  "ZINB",     "zinb",               "zinb_bamlss",      "ZINBI",     "ZINBLSS"
), ncol = 4 + 1, byrow = TRUE)

dist_df <- as.data.frame(dist[-1,], stringsAsFactors = FALSE)
colnames(dist_df) <- dist[1,]

fit_all_models <- function(formula_list, 
                           data, 
                           newdata,
                           distribution = c("normal",
                                            "lognormal",
                                            "negbin",
                                            "beta",
                                            "gamma",
                                            "logistic"),
                           args_deepregression = NULL,
                           args_bamlss = NULL,
                           args_gamlss = NULL,
                           args_mboostlss = NULL,
                           nr_folds_cv = 5,
                           iterations = 2500,
                           verbose = TRUE,
                           use_cv_dr = TRUE)
{
  
  # set distribution
  distribution <- match.arg(distribution)
  this_family <- dist_df[which(dist_df$Family == distribution),]
  
  log_score_fun <- switch(this_family$Family,
                          normal = function(y,m,s) 
                            dnorm(y,m,s,log=T),
                          gamma = function(y,m,s)
                              dGA(y,m,s,log=T),
                          logistic = function(y,m,s) 
                            dlogis(y,m,s,log=T))
  
  # define formulae
  forms <- formula_generator(formula_list, this_family)
  
  # create return list
  return_list <- list()
  
  if(verbose) cat("Fitting deepregression...\n")
  #########################################################################
  ################# Deep Distributional Regression ########################
  
  args_list <- c(list(list_of_formulae = forms[["deepregression"]], 
                      y = data$y, data = data,
                      list_of_deep_models = list(),
                      family = this_family[["deepregression"]]),
                 args_deepregression)
  
  if(this_family$Family=="gamma") args_list <- c(args_list, list(learning_rate=0.001))
  
  start_time <- Sys.time()
  mod <- do.call("deepregression", args_list)
  if(use_cv_dr){                          
    cvr_ddr <- mod %>% deepregression::cv(cv_folds = nr_folds_cv, 
                                          epochs = iterations,
                                          print_folds = FALSE,
                                          plot = FALSE)
    print(stop_here <- stop_iter_cv_result(cvr_ddr))
  }else{
    stop_here <- iterations
  }
  mod %>% fit(epochs = stop_here,
              view_metrics = FALSE,
              validation_split = NULL)
  time_elapsed <- difftime(Sys.time(), start_time, units = "sec")
  if(verbose) cat("Done in", round(time_elapsed, 2), "seconds.\n")
  
  ##### Extract results
  
  this_dist <- mod %>% get_distribution(newdata)
  
  logscore <- mean(
    do.call(log_score_fun, list(newdata$y, 
                                as.matrix(this_dist %>% tfd_mean()),
                                as.matrix(this_dist %>% tfd_stddev())))
  )
  coef <- mod %>% coef()
  plotdata <- lapply(1:2, function(i) mod %>% plot(plot=FALSE, which_param=i))
  return_list$deepregression <- list(logscore = logscore,
                                     coef = coef,
                                     plotdata = plotdata,
                                     runtime = time_elapsed)
  
  ##### Clean up
  
  rm(args_list, mod, cvr_ddr, logscore, coef, plotdata)
  gc()

  if(verbose) cat("Fitting bamlss...\n")
  #########################################################################
  ############################### BAMLSS ##################################
  
  args_list <- c(list(formula = forms[["bamlss"]], 
                      data = data,
                      family = this_family[["bamlss"]]),
                 args_bamlss)
  
  if(this_family$Family=="logistic"){
    args_list <- c(args_list, list(optimizer=boost,
                                   maxit=1000))
  }
  
  start_time <- Sys.time()
  mod <- try(do.call("bamlss", args_list))
  if(class(mod)[1]=="try-error"){ 
    
    return_list$bamlss <- list(
      logscore = NA,
      coef = NA,
      plotdata = NA,
      runtime = NA
    )
  
  }else{
    
    time_elapsed <- difftime(Sys.time(), start_time, units = "sec")
    if(verbose) cat("Done in", round(time_elapsed, 2), "seconds.\n")
    
    ##### Extract results
    
    logscore <- mean(
      do.call(log_score_fun, c(list(newdata$y), 
                               unname(mod %>% predict(newdata = newdata,
                                                      type="parameter"))[1:2]))
    )
    coef <- if(is.list(mod$parameters)) 
      lapply(mod$parameters, "[[", "p") else
        coef(mod)[,1][grepl("\\.p\\.", names(coef(mod)[,1])) & 
                        !grepl("alpha", names(coef(mod)[,1]))]
    plotdata <- lapply(c("mu","sigma"), function(w) 
      sapply(names(mod$x[[w]]$smooth.construct), function(term)
        bamlss:::predict.bamlss(mod, term = term)[w]))
    return_list$bamlss <- list(logscore = logscore,
                               coef = coef,
                               plotdata = plotdata,
                               runtime = time_elapsed)
  
  ##### Clean up
  
    rm(args_list, mod, logscore, coef)
    gc()

  }    
  
  if(verbose) cat("Fitting gamlss...\n")
  #########################################################################
  ############################### GAMLSS ##################################
  
  args_list <- c(forms[["gamlss"]],
                 list(data = data,
                      family = char_to_fun(this_family[["gamlss"]])),
                 args_gamlss)
  
  start_time <- Sys.time()
  mod <- do.call("gamlss", args_list)
  time_elapsed <- difftime(Sys.time(), start_time, units = "sec")
  if(verbose) cat("Done in", round(time_elapsed, 2), "seconds.\n")
  
  ##### Extract results
  
  logscore <-  mean(
    do.call(log_score_fun,
            unname(mod %>% predictAll(newdata = newdata, type="response"))[c(3,1:2)])
  )
  coef <- lapply(c("mu","sigma"), function(w) mod %>% coef(what = w))
  plotdata <- mod[paste0(c("mu","sigma"),".s")]
  return_list$gamlss <- list(logscore = logscore,
                             coef = coef,
                             plotdata = plotdata,
                             runtime = time_elapsed)
  
  ##### Clean up
  
  rm(args_list, mod, logscore, coef, plotdata)
  gc()
  
  
  if(verbose) cat("Fitting mboostLSS...\n")
  #########################################################################
  ############################## mboostLSS ################################
  
  fam <- if(this_family$Family%in%c("logistic","lognormal"))
            eval(parse(text=this_family[["mboostLSS"]])) else 
              char_to_fun(this_family[["mboostLSS"]])
  
  if(this_family[["Family"]]=="beta")
    names(forms[["mboostlss"]])[2] <- "phi"
  
  args_list <- c(list(formula = forms[["mboostlss"]],
                      data = data,
                      families = fam,
                      control = boost_control(
                        mstop = iterations
                      )),
                 args_mboostlss
                 )
  
  start_time <- Sys.time()
  mod <- do.call("gamboostLSS", args_list)

  if(use_cv_dr){
    cvr_mb <- cvrisk.mboostLSS(mod, 
                               folds = mboost::cv(model.weights(mod), 
                                                  B = nr_folds_cv,
                                                  type = "kfold")
    )
    mod <- mod[mstop(cvr_mb)]
  }
  time_elapsed <- difftime(Sys.time(), start_time, units = "sec")
  if(verbose) cat("Done in", round(time_elapsed, 2), "seconds.\n")
  
  ##### Extract results

  pred_param <- unname(mod %>% predict(newdata = newdata))  
  if(this_family$Family=="gamma") pred_param <- lapply(pred_param,exp)
  logscore <- mean(
    do.call(log_score_fun, c(list(newdata$y), pred_param))
  )
  
  linterms <- lapply(names(mod), function(par){
    xx <- names(mod[[par]]$baselearner)
    (1:length(mod[[par]]$baselearner))[grep("bols", xx)]
  })
  names(linterms) <- names(mod)
  sterms <- lapply(names(mod), function(par){
    xx <- names(mod[[par]]$baselearner)
    (1:length(mod[[par]]$baselearner))[grep("bbs", xx)]
  })
  names(sterms) <- names(mod)
  coef <- lapply(names(mod), function(par)
    c(mod[[par]]$offset,
      unlist(sapply(linterms[[par]], function(w) coef(mod[[par]], which=w)))))
  plotdata <- lapply(names(mod), function(par)
    lapply(sterms[[par]], function(w) predict(mod, which = w)[[par]][,1]))
  return_list$mboostLSS <- list(logscore = logscore,
                                coef = coef,
                                plotdata = plotdata,
                                runtime = time_elapsed)
  
  ##### Clean up
  
  rm(args_list, mod, cvr_mb, logscore, coef)
  gc()
  
  #########################################################################
  #########################################################################
  #########################################################################
  #########################################################################
  #########################################################################
  
  # Return everything
  return(return_list)
  
}

# function to return the mean integrated squared error for functions
# the RMSE for coefficients and predictions
calculate_results <- function(all_fits, data)
{
 
  # true coefficients and deviation functions 
  true_coef1 <- c(data$bias1,
                  data$lincoef1)
  true_coef2 <- c(data$bias2,
                  data$lincoef2)
  
  true_coef_rmse_fun <- function(pred,truth){
    sqrt(mean((pred-truth)^2))
  }
  
  true_nl1 <- data$nlfun1[data$train,]
  true_nl2 <- data$nlfun2[data$train,]
  
  # COLSUM <- function(x) if(is.null(dim(x))) sum(x) else colSums(x)
  
  true_nl_rmise_fun <- function(predMat,trueMat){
    sqrt(mean(c((predMat - trueMat)^2)))
  }
  
  # y_test <- data$y[data$test]
  # 
  # rmse_pred <- function(pred) sqrt(mean((y_test-pred)^2))
  
  ################### RMISE #################
  rmise1_deepregression <- true_nl_rmise_fun(
    sapply(all_fits$deepregression$plotdata[[1]], "[[", "partial_effect"),
    true_nl1)
  rmise1_bamlss <- true_nl_rmise_fun(
    do.call("cbind",all_fits$bamlss$plotdata[[1]]), 
    true_nl1)
  rmise1_gamlss <- true_nl_rmise_fun(
    all_fits$gamlss$plotdata[[1]], 
    true_nl1)
  rmise1_mboostLSS <- true_nl_rmise_fun(
    do.call("cbind", lapply(all_fits$mboostLSS$plotdata[[1]], 
                            function(x) if(length(x)==1) 
                              rep(x, NROW(true_nl1)) else
                              x)), 
    true_nl1)
  
  if(length(data$ind_nl2)>0){
    rmise2_deepregression <- true_nl_rmise_fun(
      sapply(all_fits$deepregression$plotdata[[2]], "[[", "partial_effect"),
      true_nl2)
    rmise2_bamlss <- true_nl_rmise_fun(
      do.call("cbind",all_fits$bamlss$plotdata[[2]]), 
      true_nl2)
    rmise2_gamlss <- true_nl_rmise_fun(
      all_fits$gamlss$plotdata[[2]], 
      true_nl2)
    rmise2_mboostLSS <- true_nl_rmise_fun(
      do.call("cbind", lapply(all_fits$mboostLSS$plotdata[[2]], 
                              function(x) if(length(x)==1) 
                                rep(x, NROW(true_nl2)) else
                                x)), 
      true_nl2)
  }else{
    rmise2_deepregression <- rmise2_bamlss <- 
      rmise2_gamlss <- rmise2_mboostLSS <- NA
  }
  
  ################### RMSE COEF #################
  rmse1_coef_deepregression <- true_coef_rmse_fun(
    all_fits$deepregression$coef[[1]][[1]][,1][1:length(true_coef1)],
    true_coef1)
  rmse1_coef_bamlss <- true_coef_rmse_fun(
    all_fits$bamlss$coef[[1]],
    true_coef1)
  rmse1_coef_gamlss <- true_coef_rmse_fun(
    all_fits$gamlss$coef[[1]][1:length(true_coef1)],
    true_coef1)
  rmse1_coef_mboostLSS <- true_coef_rmse_fun(
    all_fits$mboostLSS$coef[[1]],
    true_coef1)
  
  rmse2_coef_deepregression <- true_coef_rmse_fun(
    all_fits$deepregression$coef[[2]][[1]][,1][1:length(true_coef2)],
    true_coef2)
  rmse2_coef_bamlss <- true_coef_rmse_fun(
    all_fits$bamlss$coef[[2]],
    true_coef2)
  rmse2_coef_gamlss <- true_coef_rmse_fun(
    all_fits$gamlss$coef[[2]][1:length(true_coef2)],
    true_coef2)
  rmse2_coef_mboostLSS <- true_coef_rmse_fun(
    all_fits$mboostLSS$coef[[2]],
    true_coef2)
  
  ################### RMSE PRED #################
  # rmse_pred_deepregression <- rmse_pred(all_fits$deepregression$pred[,1])
  # rmse_pred_bamlss <- rmse_pred(all_fits$mboostLSS$pred[[1]])
  # rmse_pred_gamlss <- rmse_pred(all_fits$gamlss$pred)
  # rmse_pred_mboostLSS <- rmse_pred(all_fits$mboostLSS$pred[[1]][,1])
  
  return(data.frame(rmise_par1 = c(rmise1_deepregression, rmise1_bamlss, 
                                  rmise1_gamlss, rmise1_mboostLSS),
                    rmise_par2 = c(rmise2_deepregression, rmise2_bamlss, 
                                  rmise2_gamlss, rmise2_mboostLSS),
                    rmse_coef_par1 = c(rmse1_coef_deepregression,
                                       rmse1_coef_bamlss,
                                       rmse1_coef_gamlss,
                                       rmse1_coef_mboostLSS),
                    rmse_coef_par2 = c(rmse2_coef_deepregression,
                                       rmse2_coef_bamlss,
                                       rmse2_coef_gamlss,
                                       rmse2_coef_mboostLSS),
                    log_scores = c(all_fits$deepregression$logscore,
                                   all_fits$bamlss$logscore,
                                   all_fits$gamlss$logscore,
                                   all_fits$mboostLSS$logscore),
                    runtimes = sapply(all_fits, "[[", "runtime"),
                    method = c("deepregression", "bamlss", 
                               "gamlss", "mboostLSS")
                    )
         )
  
}