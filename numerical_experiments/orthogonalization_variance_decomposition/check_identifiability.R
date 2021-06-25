library(deepregression)

if(!dir.exists("results"))
  dir.create("results")

set.seed(24)

# generate the data
n <- 1500
maxEpochs <- 2000
p <- c(10)
b0 <- 1
simnr <- 10
rbern <- function(n, linpred) rbinom(n, 1, plogis(linpred))
rnorm_01 <- function(n, linpred) rnorm(n, linpred, 0.1)
rnorm_1 <- function(n, linpred) rnorm(n, linpred, 1)
rnorm_10 <- function(n, linpred) rnorm(n, linpred, 10)
rpoisson <- function(n, linpred) rpois(n, exp(linpred))
true_mean_linkfun <- list(rnorm_01,
                          rnorm_1,
                          rnorm_10,
                          rpoisson,
                          rbern)
family_deepregression <- c(rep("normal",3),
                           "poisson",
                           "bernoulli")
otl <- deepregression:::orthog_structured

#####################################################################
# Define a Deep Model
# We use three hidden layers for the location:
deep_model <- function(x) x %>% 
  layer_dense(units = 32, activation = "relu", use_bias = FALSE) %>% 
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 16, activation = "relu") %>% 
  layer_dense(units = 1, activation = "linear")

#####################################################################
list_of_funs <-  list(function(x) cos(5*x),
                      function(x) tanh(3*x),
                      function(x) -x^3,
                      function(x) cos(x*3-2)*(-x*3),
                      function(x) exp(x*2) - 1,
                      function(x) x^2,
                      function(x) sin(x)*cos(x),
                      function(x) sqrt(abs(x)),
                      function(x) -x^5,
                      function(x) log(abs(x)^2)/100
)

for(i in 1:length(true_mean_linkfun)){
  
  cat("Working on setting ", i, "...\n")
  
  this_tmlf <- true_mean_linkfun[[i]]
  this_family <- family_deepregression[i]
  
  res <- lapply(1:simnr, function(sim){
      
    set.seed(sim)
    # training data; predictor 
    X <- matrix(runif(p*n, -1, 1), ncol=p)
    X <- scale(X, scale=F)
    partpred_l <- sapply(1:p, function(j) 2/j*X[,j])
    interact <- scale(log(matrix(apply(X+2, 1, prod),ncol=1), base = 10), scale=F)
    partpred_nl <- sapply(1:p, function(j)
      #otl(
        otl(list_of_funs[[j]](X[,j]),cbind(matrix(rep(1,nrow(X)),ncol=1),
                                             X[,j],interact))#,
          )#)
    
    true_linpred <- b0 + 
      rowSums(partpred_l) + 
      rowSums(partpred_nl) + 
      interact
    
    # training data
    y <- this_tmlf(n, true_linpred)
    
    data = data.frame(X)
    colnames(data) <- paste0("V", 1:p)
    
    #####################################################################
    vars <- paste0("V", 1:p)
    form <- paste0("~ 1 + ", paste(vars, collapse = " + "), " + s(",
                   paste(vars, collapse = ") + s("), ") + d(",
                   paste(vars, collapse = ", "), ")")
    
    #####################################################################
    # Initialize the model using the function
    # provided in deepregression
    mod <- deepregression(
      # supply data (response and data.frame for covariates)
      y = as.numeric(y),
      data = data,
      # define how parameters should be modeled
      list_of_formulae = list(as.formula(form), ~1),
      list_of_deep_models = list(deep_model),
      learning_rate = 0.01,
      family = this_family,
      tf_seed = 1L
    )
    
    mod %>% fit(epochs = maxEpochs, 
                patience = 20, 
                verbose = FALSE,
                # early_stopping = TRUE,
                view_metrics=FALSE,
                validation_split = NULL
                )
    plotdata <- mod %>% plot(plot=FALSE)
    coef <- mod %>% coef()
    mean <- mod %>% fitted()
    lincoef <- coef[[1]][1:(p+1)]
    
    extmat <- function(char) 
      as.matrix(as.data.frame(sapply(plotdata,"[[",char)))
    
    # par(mfrow=c(1,2))
    # 
    # matplot(extmat("value"),partpred_nl, pch="o")
    # matplot(extmat("value"),extmat("partial_effect"), add=T, pch="-")
    # 
    # matplot(extmat("value"),partpred_l, pch="o")
    # matplot(extmat("value"),sapply(1:ncol(X), 
    #                                function(j) lincoef[-1][j]*X[,j]), 
    #         add=T, pch="-")
    # 
    
    
    return(list(plotdata = extmat("partial_effect"),
                plotval = extmat("value"),
                true_nl = partpred_nl,
                lin_coef = lincoef,
                sd = if(i<4) exp(as.numeric(coef[[2]][[1]])) else NULL,
                setting = i,
                true_coef = c(b0, 2/(1:p)),
                mean = mean[,1],
                true_mean = true_linpred
    ))
  })
  
  
  saveRDS(res, paste0("results/ident_family_", i, ".RDS"))
  
}

lf <- list.files("results/", full.names = T)

# first only measurements
res <- do.call("rbind", lapply(1:length(lf), function(i){
  rf <- readRDS(lf[i])

  mise_nl <- apply(sapply(rf, function(x){
    colSums((x$plotdata - x$true_nl)^2)
  }), 1, mean)

  mse_coef <- apply(sapply(rf, function(x) (x$lin_coef - x$true_coef)^2),
                    1, mean)

  if(i %in% 1:3){
    true_sd <- c(0.1,1,10)[i]
    mse_sd <- mean(sapply(rf, function(x) (x$sd - true_sd)^2))
  }else{
    mse_sd <- 0
  }

  mise_mean <- mean(colSums(sapply(rf, function(x) (x$mean - x$true_mean[,1])^2)))

  df <- data.frame(t(c(mise_nl, mse_coef, mse_sd, mise_mean, i)))
  names(df) <- c(paste0("reliMSE_function_",1:10),
                 paste0("MSE_coef_", 1:11),
                 "MSE_sd", "MSE_fit", "setting")
  return(df)

  }))

round(apply((t(sqrt(res[,grep("MSE_coef", colnames(res))])))[2:11,],
            2, function(x) c(mean(x),stats::sd(x))),4)


res2 <- do.call("rbind", lapply(1:length(lf), function(i){
  rf <- readRDS(lf[i])

  data <- do.call("rbind", lapply(1:length(rf), function(j){

    x = rf[[j]]
    data.frame(fitted = c(x$plotdata),
               truth = c(x$true_nl),
               xval = c(x$plotval),
               curve = rep(1:10, each = n),
               settings = x$setting,
               simnr = j)

  }))

}))


library(ggplot2)
library(dplyr)

res2$settings <- factor(res2$settings,
                        levels=1:5,
                        labels=c("N(sd=0.1)",
                                 "N(sd=1)",
                                 "N(sd=10)",
                                 "Poisson", "Bernoulli"))



(gg <- ggplot() +
  geom_line(data = res2 %>% filter(curve %in% c(1,2,4,5,8,9)),
            aes(x=xval, y=fitted, group=simnr), colour="grey",
            alpha=0.7, size = .8) +
  geom_smooth(data = res2 %>% filter(curve %in% c(1,2,4,5,8,9)) %>%
                filter(simnr==1),
              aes(x=xval, y=truth, group=simnr), colour="red",
            size = 0.7, linetype=2) +
  facet_grid(settings ~ curve, scales = "free_y") + theme_bw() +
  theme(strip.text.x = element_blank(),
        strip.background.x = element_blank(),
        text = element_text(size=14)) +
  xlab("feature values") + ylab("partial effect of feature")) +
  ggsave("simulation_variance_decomp_v2.pdf")




