#### Apply deep distributional regression to cancer data sets analyzed
#### in Tran et al. (2018)

rm(list=ls())
library(dplyr)
library(deepregression)

library(Metrics)
library("rmatio")
dcolon <- read.mat("VAFC_Ongetal2018/colon_train_test.mat")

deep_mod <- function(x) x %>% 
  #layer_dense(units = 16, use_bias = FALSE) %>% 
  layer_dense(units = 4) %>% 
  layer_dense(units = 1, activation = "linear")

###############################################################
# Colon

train <- dcolon$Xtr[,-1]
colnames(train) <- paste0("X", 1:2000)
test <- dcolon$Xts[,-1]
colnames(test) <- paste0("X", 1:2000)
trout <- 1*(dcolon$Ytr > 0)
tsout <- 1*(dcolon$Yts > 0)

form <- as.formula(paste0("~", paste(colnames(train), collapse=" + "),
                          "+ d(", paste(colnames(train), collapse=", "), ")"))


reslist <- list()

for(i in 1:20){
  
  mod <- deepregression(y = trout,
                        data = as.data.frame(train),
                        list_of_formulae = 
                          list(logit = form),
                        family = "bernoulli",
                        lambda_lasso = NULL,#0.0,
                        lambda_ridge = NULL,#50,
                        list_of_deep_models = list(d=deep_mod),
                        cv_folds = 5)
  
  cvres <- mod %>% cv(epochs = 450)
  
  st <- Sys.time()
  mod %>% fit(epochs = #400,
              stop_iter_cv_result(cvres),
              validation_split = NULL, view_metrics=FALSE)
  dur <- as.numeric(difftime(Sys.time(),st,units = "sec"))
  
  pred <- mod %>% predict(newdata = as.data.frame(test))
  boxplot(pred ~ tsout)
  reslist[[i]] <- data.frame(pred = pred, truth = tsout, simnr = i, duration=dur)

}

res <- do.call("rbind", reslist)
res %>% group_by(simnr) %>% summarize(auc=auc(truth, pred)) %>% 
  summarise(value = paste0(round(mean(auc, na.rm=T),3),
                           " (", round(sd(auc, na.rm=T),3), ")"))
# 1.00 (0.00)

####################################################################
# Breast

deep_mod <- function(x) x %>% 
  layer_dense(units = 16, use_bias = FALSE) %>% 
  layer_dense(units = 4) %>% 
  layer_dense(units = 1, activation = "linear")

dbreast <- read.mat("VAFC_Ongetal2018/duke_train_test.mat")

train <- dbreast$Xtr[,-1]
colnames(train) <- paste0("X", 1:7129)
test <- dbreast$Xts[,-1]
colnames(test) <- paste0("X", 1:7129)
trout <- 1*(dbreast$Ytr > 0)
tsout <- 1*(dbreast$Yts > 0)

form <- as.formula(paste0("~", paste(colnames(train), collapse=" + "),
                          "+ d(", paste(colnames(train), collapse=", "), ")"))

reslist <- list()

for(i in 1:20){
  
  mod <- deepregression(y = trout,
                        data = as.data.frame(train),
                        list_of_formulae = 
                          list(logit = form),
                        family = "bernoulli",
                        lambda_ridge = NULL,#0,
                        lambda_lasso = NULL,#4,
                        list_of_deep_models = list(d=deep_mod),
                        cv_folds = 3#,df=0.1
  )
  
  cvres <- mod %>% cv(epochs = 100)
  
  
  st <- Sys.time()
  mod %>% fit(epochs = #100, 
              stop_iter_cv_result(cvres),
              validation_split = NULL, view_metrics=FALSE)
  # coef <- mod %>% coef()
  # summary(coef[[1]][[1]])
  
  dur <- as.numeric(difftime(Sys.time(),st,units = "sec"))
  
  pred <- mod %>% predict(newdata = as.data.frame(test))
  boxplot(pred ~ tsout)
  reslist[[i]] <- data.frame(pred = pred, truth = tsout, simnr = i, duration=dur)
  
}

res <- do.call("rbind", reslist)
res %>% group_by(simnr) %>% summarize(auc=auc(truth, pred)) %>% 
  summarise(value = paste0(round(mean(auc, na.rm=T),3),
                           " (", round(sd(auc, na.rm=T),3), ")"))

# 1 (0)

####################################################################
# Leukaemia 

dleuk <- read.mat("VAFC_Ongetal2018/leukemia_train_test.mat")

train <- dleuk$Xtr[,-1]
colnames(train) <- paste0("X", 1:7129)
test <- dleuk$Xts[,-1]
colnames(test) <- paste0("X", 1:7129)
trout <- 1*(dleuk$Ytr > 0)
tsout <- 1*(dleuk$Yts > 0)

form <- as.formula(paste0("~", paste(colnames(train), collapse=" + "),
                          "+ d(", paste(colnames(train), collapse=", "), ")"))

deep_mod <- function(x) x %>% 
 layer_dense(units = 4, activation = "relu", use_bias = FALSE) %>%
  layer_dense(units = 2, activation = "relu") %>%
  layer_dense(units = 1, activation = "linear")

reslist <- list()

for(i in 1:20){
  
  mod <- deepregression(y = trout,
                        data = as.data.frame(train),
                        list_of_formulae = 
                          list(logit = form),
                        family = "bernoulli",
                        lambda_ridge = NULL,
                        lambda_lasso = NULL, #100,
                        list_of_deep_models = list(d=deep_mod),
                        cv_folds = 3#,
                        #df=0.1
  )
  
  
  cvres <- mod %>% cv(epochs = 100)
  
  st <- Sys.time()
  mod %>% fit(epochs = #900, 
              stop_iter_cv_result(cvres),
              validation_split = NULL, view_metrics=FALSE)
  # coef <- mod %>% coef()
  # summary(coef[[1]][[1]])
  
  dur <- as.numeric(difftime(Sys.time(),st,units = "sec"))
  
  pred <- mod %>% predict(newdata = as.data.frame(test))
  boxplot(pred ~ tsout)
  reslist[[i]] <- data.frame(pred = pred, truth = tsout, simnr = i, duration=dur)
  
}

res <- do.call("rbind", reslist)
res %>% group_by(simnr) %>% summarize(auc=auc(truth, pred)) %>% 
  summarise(value = paste0(round(mean(auc, na.rm=T),3),
                           " (", round(sd(auc, na.rm=T),3), ")"))

# 0.978 (0.017)

#############################################################################################
# Using the same network without SDDL


rm(list=ls())
library(devtools)
library(keras)

# lf <- list.files("data/highdim_LR/", full.names = T)
library(Metrics)
library("rmatio")
dcolon <- read.mat("VAFC_Ongetal2018/colon_train_test.mat")

###############################################################
# Colon


train <- dcolon$Xtr[,-1]
colnames(train) <- paste0("X", 1:2000)
test <- dcolon$Xts[,-1]
colnames(test) <- paste0("X", 1:2000)
trout <- 1*(dcolon$Ytr > 0)
tsout <- 1*(dcolon$Yts > 0)


reslist <- list()

for(i in 1:20){
  
  dnn <- keras_model_sequential()
  
  dnn %>% 
    layer_dense(units = 4) %>% 
    layer_dense(units = 1, activation = "sigmoid") %>% 
    compile(
      optimizer = "adam",
      loss      = "binary_crossentropy",
      metrics   = "accuracy"
    )
  
  # fit model
  history_keras <- fit(
    object           = dnn, 
    x                = train, 
    y                = trout,
    batch_size       = 42, 
    epochs           = 450,
    validation_split = NULL,
    view_metrics = FALSE
  )
  
  pred <- dnn %>% predict(test)
  boxplot(pred ~ tsout)
  reslist[[i]] <- data.frame(pred = pred, truth = tsout, simnr = i)
  
}

res <- do.call("rbind", reslist)
saveRDS(res, file="predictions_dnn_colon.RDS")
res %>% group_by(simnr) %>% summarize(auc=auc(truth, pred)) %>% 
  summarise(value = paste0(round(mean(auc, na.rm=T),3),
                           " (", round(sd(auc, na.rm=T),3), ")"))
# 1.00 (0.00)

####################################################################
# Breast

dbreast <- read.mat("VAFC_Ongetal2018/duke_train_test.mat")

train <- dbreast$Xtr[,-1]
colnames(train) <- paste0("X", 1:7129)
test <- dbreast$Xts[,-1]
colnames(test) <- paste0("X", 1:7129)
trout <- 1*(dbreast$Ytr > 0)
tsout <- 1*(dbreast$Yts > 0)

reslist <- list()

for(i in 1:20){
  
  dnn <- keras_model_sequential()
  
  dnn %>% 
    layer_dense(units = 16) %>% 
    layer_dense(units = 4) %>% 
    layer_dense(units = 1, activation = "sigmoid") %>% 
    compile(
      optimizer = "adam",
      loss      = "binary_crossentropy",
      metrics   = "accuracy"
    )
  
  # fit model
  history_keras <- fit(
    object           = dnn, 
    x                = train, 
    y                = trout,
    batch_size       = 38, 
    epochs           = 100,
    validation_split = NULL,
    view_metrics = FALSE
  )
  
  pred <- dnn %>% predict(test)
  boxplot(pred ~ tsout)
  reslist[[i]] <- data.frame(pred = pred, truth = tsout, simnr = i)
  
}

res <- do.call("rbind", reslist)
res %>% group_by(simnr) %>% summarize(auc=auc(truth, pred)) %>% 
  summarise(value = paste0(round(mean(auc, na.rm=T),3),
                           " (", round(sd(auc, na.rm=T),3), ")"))

# 1 (0)

####################################################################
# Leukaemia 

dleuk <- read.mat("VAFC_Ongetal2018/leukemia_train_test.mat")

train <- dleuk$Xtr[,-1]
colnames(train) <- paste0("X", 1:7129)
test <- dleuk$Xts[,-1]
colnames(test) <- paste0("X", 1:7129)
trout <- 1*(dleuk$Ytr > 0)
tsout <- 1*(dleuk$Yts > 0)


reslist <- list()

for(i in 1:20){
  
  dnn <- keras_model_sequential()
  
  dnn %>% 
    layer_dense(units = 4, activation = "relu") %>%
    layer_dense(units = 2, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid") %>% 
    compile(
      optimizer = "adam",
      loss      = "binary_crossentropy",
      metrics   = "accuracy"
    )
  
  # fit model
  history_keras <- fit(
    object           = dnn, 
    x                = train, 
    y                = trout,
    batch_size       = 38, 
    epochs           = 100,
    validation_split = NULL,
    view_metrics = FALSE
  )
  
  pred <- dnn %>% predict(test)
  boxplot(pred ~ tsout)
  reslist[[i]] <- data.frame(pred = pred, truth = tsout, simnr = i)
  
}

res <- do.call("rbind", reslist)
res %>% group_by(simnr) %>% summarize(auc=auc(truth, pred)) %>% 
  summarise(value = paste0(round(mean(auc, na.rm=T),3),
                           " (", round(sd(auc, na.rm=T),3), ")"))
# 0.824 (0.234)
