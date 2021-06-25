library(deepregression)

## ------------------------------------------------------------------------------------
## simulate data

settings <- expand.grid(replication = 1:10,
                        n = c(100, 1000, 10000),
                        p = c(1, 10))
set.seed(42)
toyX = matrix(rnorm(max(settings$n)*max(settings$p)), ncol = max(settings$p))
colnames(toyX) <- paste0("x", 1:max(settings$p))
betas = seq(-3, 3, l = max(settings$p))

dgp <- function(i)
{
  
  set <- settings[i,,drop=F]
  this_X <- toyX[1:set$n, 1:set$p, drop=F]
  set.seed(set$replication)
  noise <- rnorm(set$n)
  this_y <- this_X%*%betas[1:set$p] + noise
  return(list(X=this_X, y=this_y))
  
}

deep_model <- function(x)
{
  x %>% 
    layer_dense(units = 100, activation = "relu", use_bias = FALSE) %>%
    layer_dense(units = 50, activation = "relu") %>%
    layer_dense(units = 1, activation = "linear")
}


## ------------------------------------------------------------------------------------
## simulation

sim_res <- list()

for(i in 1:nrow(settings)){
  
  p <- settings[i,"p"]
  xs <- paste(paste0("x", 1:p), collapse=" + ")
  xu <- paste(paste0("x", 1:p), collapse=", ")
  
  forms <- list(loc = as.formula(
    paste("~ -1 + ", xs, " + ", "deep_model(", xu, ")")
    ), 
    scale = ~ 1)
  
  data = dgp(i)
  
  args <- list(
    y = data$y, 
    data = as.data.frame(data$X), 
    list_of_formulae = forms, 
    list_of_deep_models = list(deep_model = deep_model),
    tf_seed = 1L
  )
  
  mod_w_oz <- do.call("deepregression", c(args, list(orthogonalize = TRUE)))
  mod_w_oz_25 <- do.call("deepregression", c(args, list(orthogonalize = TRUE)))
  mod_w_oz_50 <- do.call("deepregression", c(args, list(orthogonalize = TRUE)))
  mod_wo_oz <- do.call("deepregression", c(args, list(orthogonalize = FALSE)))
  mod_wo_oz_25 <- do.call("deepregression", c(args, list(orthogonalize = FALSE)))
  mod_wo_oz_50 <- do.call("deepregression", c(args, list(orthogonalize = FALSE)))
  
  mod_w_oz %>% fit(epochs = 1000, early_stopping = TRUE, verbose = FALSE, batch_size = as.integer(settings[i,"n"]))
  mod_wo_oz %>% fit(epochs = 1000, early_stopping = TRUE, verbose = FALSE, batch_size = as.integer(settings[i,"n"]))
  mod_w_oz_25 %>% fit(epochs = 1000, early_stopping = TRUE, verbose = FALSE, batch_size = 25)
  mod_wo_oz_25 %>% fit(epochs = 1000, early_stopping = TRUE, verbose = FALSE, batch_size = 25)
  mod_w_oz_50 %>% fit(epochs = 1000, early_stopping = TRUE, verbose = FALSE, batch_size = 50)
  mod_wo_oz_50 %>% fit(epochs = 1000, early_stopping = TRUE, verbose = FALSE, batch_size = 50)

  sim_res[[i]] <- cbind(
    settings[i,,drop=F],
    data.frame(
      with = coef(mod_w_oz, params = 1)[[1]],
      without = coef(mod_wo_oz, params = 1)[[1]],
      with_25 = coef(mod_w_oz_25, params = 1)[[1]],
      without_25 = coef(mod_wo_oz_25, params = 1)[[1]],
      with_50 = coef(mod_w_oz_50, params = 1)[[1]],
      without_50 = coef(mod_wo_oz_50, params = 1)[[1]],
      linmod = coef(lm(data$y ~ 0 + data$X))
    )
  )
  
}

saveRDS(sim_res, file="simulation_oz.RDS")
res <- do.call("rbind", sim_res)

library(dplyr)
library(reshape2)
library(ggplot2)
library(ggsci)

res[,4:9] <- res[,4:9]-res[,10]


resf <- cbind(res[,c(1:3,4:5)], batch_size="full")
res25 <- cbind(res[,c(1:3,6:7)], batch_size="25")
colnames(res25)[4:5] <- c("with", "without")
res50 <-  cbind(res[,c(1:3,8:9)], batch_size="50")
colnames(res50)[4:5] <- c("with", "without")

res <- rbind(resf, res25, res50)

res$with <- res$with^2
res$without <- res$without^2

res <- melt(res, id.vars=c("replication", "n", "p", "batch_size"))

res %>% group_by(n, p, batch_size, variable) %>% 
  summarise(mse = paste0(round(mean(value),4)," (", round(stats::sd(value), 4),")"))

levels(res$variable) <- c("with Orthogonalization", "without Orthogonalization")
res$p <- factor(res$p, levels = c(1,10), labels = c("p = 1", "p = 10"))
res$n <- as.factor(res$n)

res %>% ggplot(aes(x = n, y = value, fill = batch_size)) + 
  facet_grid(p~variable) + geom_boxplot(position = "dodge") + theme_bw() + 
  xlab("Number of observations n") + 
  ylab("Squared deviation") + scale_fill_jama(name = "Batch size") + 
  ggsave(file = "batch_oz.pdf", width = 8, height = 5)
