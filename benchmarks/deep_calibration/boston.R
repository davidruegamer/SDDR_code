# load data saved in python with specific random_state

y_train <- read.csv("data/boston/y_train.csv", header=F)
y_test <- read.csv("data/boston/y_test.csv", header=F)
x_train <- read.csv("data/boston/x_train.csv", header=F)
x_test <- read.csv("data/boston/x_test.csv", header=F)

# load deepregression
library(deepregression)

set.seed(42)

# define measures

res = data.frame(LL = NA, MSE = NA, time = NA)

nrsims <- 20

Vs <- paste0("V",1:13)
form_mu <- paste0("~ 1 ", 
                  "+",
                  paste(Vs, collapse=" + "),
                  " + s(",
                  paste(Vs[c(-4,-9)], collapse=") + s("), ")",
                  "+ deep_mu(",
                  paste(Vs, collapse=", "), ")")

form_sig <- paste0("~ 1", 
                   "+",
                   "deep_sig(",
                  paste(Vs, collapse=", "), ")"
                  )

deep_mod <- function(x) x %>% 
  layer_dense(units = 32, activation = "tanh", use_bias = FALSE) %>%
  layer_dense(units = 16, activation = "tanh") %>% 
  layer_dense(units = 4, activation = "tanh") %>% 
  layer_dense(units = 1, activation = "linear")

deep_mod2 <- function(x) x %>% 
  layer_dense(units = 2, activation = "tanh", use_bias = FALSE) %>%
  layer_dense(units = 1, activation = "linear")


for(sim_iteration in 1:nrsims){
  
  mod_deep <- deepregression(y = y_train$V1, 
                             list_of_formulae = list(loc = as.formula(form_mu),
                                                     scale = as.formula(form_sig)),
                             list_of_deep_models = list(deep_mu=deep_mod, 
                                                        deep_sig=deep_mod2),
                             data = x_train,
                             family = "normal",
                             cv_folds = 5)
  
  st <- Sys.time()
  
  cvres <- mod_deep %>% cv(epochs = 1000)
  
  (ep <- stop_iter_cv_result(cvres))
  
  mod_deep %>% fit(epochs = ep, 
                   verbose = FALSE, view_metrics = FALSE,
                   validation_split = NULL)
  
  et <- Sys.time()
  
  pred <- mod_deep %>% predict(x_test)
  this_dist <- mod_deep %>% get_distribution(x_test, force_float = T)
  
  log_score_fun <- function(y,m,s) dnorm(y,m,s,log=T)
  
  (ll <- -mean(
    do.call(log_score_fun, list(y_test$V1, 
                                as.matrix(this_dist %>% tfd_mean()),
                                as.matrix(this_dist %>% tfd_stddev())))
  ))
  
  (mse <- (mean((pred-y_test$V1)^2)))
  
  res[sim_iteration, ] <- c(ll, mse, as.numeric(difftime(et,st,units="mins")))
  
}

# get performance and times
apply(res, 2, function(x) c(mean(x, na.rm=T), sd(x, na.rm=T)))


# LL       MSE      time
# [1,] 3.142710 19.410764 3.7448835
# [2,] 0.290823  1.756809 0.1134728