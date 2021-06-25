#### Apply deep distributional regression to 
#### Cornwell and Rupert Returns to Schooling Data, 
#### longitud. study, 595 Individuals, 7 Years
#### also used in Tran et al. (2018)
#### for within subject prediction

rm(list=ls())

library(tidyverse)
library(deepregression)

set.seed(48)

data <- read.csv("http://people.stern.nyu.edu/wgreene/Econometrics/cornwell&rupert.csv")
data$ID <- as.factor(data$ID)

train <- data %>% filter(YEAR < 6)
test <- data %>% filter(YEAR >= 6)

deep_mod_m <- function(x) x %>% 
  layer_dense(units = 5, activation = "relu", use_bias = FALSE) %>%
  layer_dense(units = 5, activation = "relu") %>%
  layer_dense(units = 1, activation = "linear")

deep_mod_s <- function(x) x %>% 
  layer_dense(units = 5, activation = "relu", use_bias = TRUE) %>%
  layer_dense(units = 5, activation = "relu") %>%
  layer_dense(units = 1, activation = "linear")

# expanding window CV
cv_folds <- list(#year1 = list(train = which(train$YEAR==1),
                #              test = which(train$YEAR>1 & train$YEAR<4)),
                 year2 = list(train = which(train$YEAR<=2),
                              test = which(train$YEAR>2 & train$YEAR<5)),
                 year3 = list(train = which(train$YEAR<=3),
                              test = which(train$YEAR>3 & train$YEAR<6)))

simnr <- 20

mses <- rep(NA,20)
pps <- rep(NA,20)

for(sn in 1:simnr){
  
  f1 <- ~ 1 + s(ID, bs="re") + #s(EXP) + s(WKS) + 
    deep_mod_m(YEAR, EXP, WKS, OCC, IND, SOUTH, SMSA, MS, FEM, UNION, ED, BLK)
  f2 <- ~1 #+ s(EXP) + s(WKS) #+ 
    #deep_mod_s(YEAR, EXP, WKS, OCC, IND, SOUTH, SMSA, MS, FEM, UNION, ED, BLK)
  
  # initialize model
  mod <- deepregression(y = train$LWAGE,
                        data = train[,c(1:11, 14, 16)], 
                        list_of_formulae = list(f1, f2),
                        list_of_deep_models = list(deep_mod_m = deep_mod_m,
                                                   deep_mod_s = deep_mod_s),
                        family = "normal",
                        cv_folds = cv_folds
  )
  
  cvres <- mod %>% cv(epochs = 400)
  mod %>% fit(epochs = stop_iter_cv_result(cvres), view_metrics=FALSE)
  pred <- mod %>% predict(test)
  sd <- (mod %>% sd.deepregression(test))
  
  (mses[sn] <- mean((pred-test$LWAGE)^2))
  (pps[sn] <- - mean(dnorm(x = test$LWAGE, mean = pred, sd = sd, log = T)))
  
}

mean(mses)
sd(mses)
mean(pps)
sd(pps)
