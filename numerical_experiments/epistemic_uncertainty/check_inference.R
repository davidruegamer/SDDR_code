rm(list=ls())
library(deepregression)
nrCores <- 1
load_all(path)
library(parallel)

if(!dir.exists("results"))
  dir.create("results")

# function definitions
collapseS <- function(x) paste0("s(", paste(x, collapse = ") + s("), ")")
collapseD <- function(x) paste0("d(", paste(x, collapse = ","), ")")
center_nl <- function(fun){
  deepregression:::orthog_structured(fun, matrix(rep(1,length(fun)),
                                                 ncol=1))
}
functions = list(function(x) sin(x)*cos(x),
                 function(x) -x^5,
                 function(x) x^2,
                 function(x) (sin(10*x)*0.1 + 1) * (x<0) + (-2*x + 1) * (x >=0),
                 function(x) -x * tanh(3*x) * sin(4*x),
                 function(x) sin(10*x)
)

# create data
n <- 5000
klw <<- 1 / n
alpha = 0.1
qa <- -qnorm(alpha/2)
maxEpochs <- 1000
simiter = 20

# settings
settings <- expand.grid(number_nl_effects = c(1,3,5),
                        deep = c(FALSE, TRUE),
                        variational = c(FALSE, TRUE))

X <- matrix(runif(n * (5 + 5), -1, 1), nrow = n)
colnames(X) <- paste0("V", 1:ncol(X))
X <- scale(X, scale = F)

true_nl_1 <- center_nl(functions[[1]](X[,1]))
true_nl_2 <- center_nl(functions[[2]](X[,2]))
true_nl_3 <- center_nl(functions[[3]](X[,3]))
true_nl_4 <- center_nl(functions[[4]](X[,4]))
true_nl_5 <- center_nl(functions[[5]](X[,5]))

true_nl <- cbind(true_nl_1,
                 true_nl_2,
                 true_nl_3,
                 true_nl_4,
                 true_nl_5)

b <- 0



for(i in 1:nrow(settings)){
  
  cat("Working on setting ", i, " ...\n")
  
  set <- settings[i,]
  
  eta <- #exp( 
    b + apply(true_nl[,1:set$number_nl_effects,drop=F],1,sum)
  #)

  deep_model <- function(x) x %>% 
    layer_dense(units = 16, activation = "relu") %>% 
    layer_dense(units = 1, activation = "linear")  
  
  Xval <- X <- as.data.frame(X)
  etaval <- eta
  
  form <- vector("list", 2)
  
  form[[1]] <- paste0("~1 + ", paste(
    paste0("s(V", 1:set$number_nl_effects, ")"), collapse=" + "))
  
  if(set$deep){
    form[[1]] <- paste0(form[[1]], "+ deep_model(V6,V7,V8,V9,V10)")
    lod <- list(deep_model=deep_model)
  }else{
    lod <- list(NULL)
  }
  
  form[[2]] <- "~1"
  
  res <- mclapply(1:simiter, function(j){
    
    st <- Sys.time()
    
    y <- rnorm(n, mean = eta, sd = sqrt(var(eta)))
    data = cbind(data.frame(y=y), X)
    dataval = list(Xval, etaval)
    
    mod <- deepregression(y, 
                          list_of_formulae = lapply(form, as.formula), # just use the intercept
                          data = data,
                          list_of_deep_models = lod, 
                          family = "normal",
                          variational = set$variational, 
                          df = 4,
                          # learning_rate = 0.1,
                          #validation_data = dataval#,
                          validation_split = NULL#,
                          #optimizer = keras:::optimizer_rmsprop()#,
                          # prior_fun = function(kernel_size, bias_size, dtype) 
                          #   prior_trainable(kernel_size,
                          #                   bias_size = 0,
                          #                   dtype = NULL,
                          #                   diffuse_scale = 100)
                          )
    
    
    mod %>% fit(epochs = maxEpochs, view_metrics=FALSE,
                validation_split = NULL
                #, early_stopping = F,#TRUE,
                #patience = 10, verbose=TRUE
                )
    plotData <- mod %>% plot(which_param = 1, 
                             use_posterior = set$variational, 
                             plot=FALSE)
    if(class(plotData)[1]=="try-error") return(NULL)
    print(Sys.time()-st)
    log_score <- mean(mod %>% log_score())
    
    # true values
    pe_true <- true_nl[,1:set$number_nl_effects,drop=F]
    
    if(!set$variational){
      
      pe_fit <- sapply(plotData, #[1:set$number_nl_effects] 
                       "[[", "partial_effect")
      
    }else{
      
      pe_fit <- sapply(plotData, #[1:set$number_nl_effects] 
                       "[[", "mean_partial_effect")
      
      pe_sd <- sapply(plotData, #[1:set$number_nl_effects], 
                         "[[", "sd_partial_effect")
      
    }
    
   sqdev_nl <- apply((pe_true - pe_fit)^2, 2, mean)
    
   pc_cov <- pc_nz <- NA
   
   if(set$variational){
     
     lo_pe <- pe_fit - qa * pe_sd
     up_pe <- pe_fit + qa * pe_sd
     
     pc_cov <- apply(lo_pe <= pe_true & up_pe >= pe_true, 2, mean)
     pc_nz <- apply(lo_pe * up_pe > 0, 2, mean)
     
     pc_cov <- c(pc_cov, rep(NA,5-length(pc_cov)))
     pc_nz <- c(pc_nz, rep(NA,5-length(pc_cov)))
     
   }
   
   results <- cbind(set, 
                    data.frame(
                      integrated_squared_dev_pe_po = t(sqdev_nl),
                      percent_coverage_pe_po = t(pc_cov),
                      percent_nonzero_pe_po = t(pc_nz)
                    ))
   
   saveRDS(results, file = paste0("results/uncertainty/setting_",
                                  i, "_iteration_", j, ".RDS"))
   
   rm(mod, results, plotData, data, dataval)
   gc()
   
   return(NULL)
   
  }, mc.cores = nrCores)
  
}
# 
lf <- list.files("results/uncertainty/", full.names = T)
# 
library(dplyr)
library(tidyr)
library(stringr)

res <- lapply(lf, readRDS)
res <- lapply(res, function(re){
  
  # if("integrated_squared_dev_pe_po" %in% colnames(re)){
  #   cre <- colnames(re)
  #   colnames(re)[which(cre=="integrated_squared_dev_pe_po")] <- 
  #     "integrated_squared_dev_pe_po.1"
  #   re <- cbind(re,t(rep(NA,4)))
  #   colnames(re)[(ncol(re)-3):ncol(re)] <- 
  #     paste0("integrated_squared_dev_pe_po.",2:5)
  # }
  # 
  # if("percent_coverage_pe_po" %in% colnames(re)){
  #   cre <- colnames(re)
  #   colnames(re)[which(cre=="percent_coverage_pe_po")] <- 
  #     "percent_coverage_pe_po.1"
  #   re <- cbind(re,t(rep(NA,4)))
  #   colnames(re)[(ncol(re)-3):ncol(re)] <- 
  #     paste0("percent_coverage_pe_po.",2:5)
  # }
  # 
  # if("percent_nonzero_pe_po" %in% colnames(re)){
  #   cre <- colnames(re)
  #   colnames(re)[which(cre=="percent_nonzero_pe_po")] <- 
  #     "percent_nonzero_pe_po.1"
  #   re <- cbind(re,t(rep(NA,4)))
  #   colnames(re)[(ncol(re)-3):ncol(re)] <- 
  #     paste0("percent_nonzero_pe_po.",2:5)
  # }
  re <- re %>% gather(measure, value, -c(deep, number_nl_effects, variational))
  re$measure[!grepl("[0-9]",re$measure)] <- paste0(re$measure[!grepl("[0-9]",re$measure)] ,".1")
  re$var <- str_sub(re$measure, -1, -1)
  re$measure <- gsub("(.+)\\_pe\\_po\\.[0-9]","\\1",re$measure)
  return(re)
  
})



#### Analysis of results

res <- do.call("rbind", res)
res <- res %>% filter(!is.na(value))

library(xtable)
library(ggplot2)
library(reshape2)

res %>% filter(measure=="integrated_squared_dev") %>% 
  group_by(#deep, => no change with deep
    variational, var) %>% 
  summarise(value = paste0(round(mean(value, na.rm=T),3),
                   " (", round(sd(value, na.rm=T),3), ")")) %>% 
  xtable()

res %>% filter(measure=="percent_coverage", variational==TRUE) %>% 
  group_by(# deep, => no change with deep 
           var) %>% 
  summarise(value = paste0(round(mean(value, na.rm=T),3),
                           " (", round(sd(value, na.rm=T),3), ")")) %>% 
  xtable()


res %>% filter(measure=="percent_nonzero", variational==TRUE) %>% 
  group_by(var,#deep, #=> no change with deep 
    #number_nl_effects
    ) %>% 
  summarise(value = paste0(round(mean(value, na.rm=T),3),
                           " (", round(sd(value, na.rm=T),3), ")")) %>% 
  xtable()


# 
# res$intsqdev <- apply(res[,grep("integrated_squared_dev_pe_po",
#                                 colnames(res))], 1, function(x) mean(x,na.rm = T))
# res$coverage <- apply(res[,grep("percent_coverage_pe_po",
#                                 colnames(res))], 1, function(x) mean(x,na.rm = T))
# res$nonzero <- apply(res[,grep("percent_nonzero_pe_po",
#                                 colnames(res))], 1, function(x) mean(x,na.rm = T))
# 
# sres <- res %>% select(deep, number_nl_effects, variational,
#                        intsqdev, coverage, nonzero) %>% 
#   group_by(deep, number_nl_effects, variational) %>%
#   summarise_all(list(~mean(.,na.rm = T),~sd(.,na.rm = T)))
# 
# sres %>% melt(., id.vars = c("deep","number_nl_effects","variational")) %>% 
#   ggplot(aes(x=variable,y=value)) + geom_point() + theme_bw() + 
#   facet_grid(number_nl_effects ~ deep*variational, labeller = label_both) + 
#   theme(axis.text.x = element_text(angle=45, hjust = 1))
# 
# sres %>% xtable()
# 
# 
