airfoil <- read.table("data/airfoil/airfoil_self_noise.dat")

set.seed(42)

index_train <- sample(1:nrow(airfoil), round(nrow(airfoil)*0.75))

train <- airfoil[index_train,]
test <- airfoil[setdiff(1:nrow(airfoil), index_train),]

write.csv(train, "data/airfoil/train.csv")
write.csv(test, "data/airfoil/test.csv")

# load deepregression
library(deepregression)

# define measures

res = data.frame(LL = NA, MSE = NA, time = NA)

nrsims <- 20

Vs <- paste0("V",1:5)
form_mu <- paste0("~ 1",
                  # "+",
                  # paste(Vs, collapse=" + "), 
                  " + s(",
                  paste(Vs[c(-3,-4)], collapse=") + s("), ") + d(",
                  paste(Vs, collapse=", "), ")")

form_sig <- paste0("~ 1 + ", "d(",
                   paste(Vs, collapse=", "), ")")

deep_mod <- function(x) x %>% 
  layer_dense(units = 16, activation = "tanh", use_bias = FALSE) %>%
  layer_dense(units = 4, activation = "tanh") %>% 
  layer_dense(units = 1, activation = "linear")

for(sim_iteration in 1:nrsims){
  
  mod_deep <- deepregression(y = train$V6, 
                             list_of_formulae = list(loc = as.formula(form_mu),
                                                     scale = as.formula(form_sig)),
                             list_of_deep_models = list(deep_mod, 
                                                        deep_mod),
                             data = train[,1:5],
                             family = "normal",
                             cv_folds = 5)
  
  st <- Sys.time()
  
  cvres <- mod_deep %>% cv(epochs = 1000)
  
  (ep <- stop_iter_cv_result(cvres))
  
  mod_deep %>% fit(epochs = ep, 
                   verbose = FALSE, view_metrics = FALSE,
                   validation_split = NULL)
  
  et <- Sys.time()
  
  pred <- mod_deep %>% predict(test[,1:5])
  this_dist <- mod_deep %>% get_distribution(test[,1:5], force_float=T)
  
  log_score_fun <- function(y,m,s) dnorm(y,m,s,log=T)
  
  (ll <- -mean(
    do.call(log_score_fun, list(test$V6, 
                                as.matrix(this_dist %>% tfd_mean()),
                                as.matrix(this_dist %>% tfd_stddev())))
  ))
  
  (mse <- (mean((pred-test$V6)^2)))
  
  res[sim_iteration, ] <- c(ll, mse, as.numeric(difftime(et,st,units="mins")))
  
}

# get performance and times
apply(res, 2, function(x) c(mean(x, na.rm=T), sd(x, na.rm=T)))
# LL       MSE     time
# [1,] 3.1145236 29.588746 14.47671
# [2,] 0.0219707  1.246679  1.01858