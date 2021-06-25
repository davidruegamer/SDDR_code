lf <- list.files("results/model_comparisons/", full.names = T)
res <- do.call("rbind", lapply(lf, function(l) readRDS(l)))

library(ggplot2)
library(dplyr)
library(reshape2)

mres <- melt(res, id.vars = names(res)[-1*c(1:6)])
mres$variable <- as.character(mres$variable)

mres$variable[mres$variable%in%c("rmise_par1","rmise_par2",
                                 "rmse_coef_par1","rmse_coef_par2")] <- "RMSE"


mres %>% group_by(method,family,n,p1,variable) %>% 
  summarize(medianv = median(value,na.rm = T)) %>% 
  filter(variable!="runtimes", medianv <= 1000) %>% 
  ggplot(., aes(x=method,y=medianv,colour = interaction(p1,n))) + geom_point() + 
  facet_grid(variable~family, scales = "free") 

library(xtable)
library(tidyr)

trmres <- mres %>% group_by(variable,method,family,n,p1) %>% 
  summarize(val = paste0(round(median(value,na.rm = T),2), 
                         " (",round(mad(value,na.rm=T),2),")")) %>% 
  filter(variable!="runtimes") %>% spread(method, val)

trmres$variable <- factor(trmres$variable,
                          levels = unique(trmres$variable),
                          c("Log-scores", "RMSE"))

trmres$family <- factor(trmres$family,
                        levels = unique(trmres$family),
                        labels = c("Normal", "Gamma", "Logistic"))

trmres %>% 
  xtable(row.names=F)

# mres$variable[mres$variable%in%c()] <- "lin. terms"
# mres$variable[mres$variable=="rmse_pred"] <- "prediction"


# summary_res <- mres %>% group_by(variable, method,
#                                  n, p1, p2, pnl1, pnl2, family) %>% 
#   summarise(agg_val = mean(value, na.rm=T)) %>% filter(family!="negbin", variable!="runtimes")

# summary_res$agg_val[summary_res$agg_val>1000] <- NA

# ggplot(summary_res %>% filter(variable != "runtimes"), 
#        aes(x= interaction(p1,p2,pnl1,pnl2), y=agg_val, 
#                         colour = method, group = method)) + 
#   geom_point(position = position_dodge(0.2)) + 
#   # geom_errorbar(aes(ymin = agg_val, ymax = agg_val + 2*sd_val),
#   #               width = 0.2,position = position_dodge(0.2)) +
#   facet_grid(variable ~ family, scales = "free_y")

# rel_res <- summary_res %>% group_by(variable, n, p1, p2, pnl1, pnl2, family) %>% 
#   mutate(value_rel = pmin(100, agg_val / min(agg_val, na.rm=T)))
# 
# ggplot(rel_res, 
#        aes(x= interaction(p1,p2,pnl1,pnl2,n), y=value_rel, 
#            colour = method, group = method)) + 
#   geom_point(position = position_dodge(0.2)) + 
#   # geom_errorbar(aes(ymin = agg_val, ymax = agg_val + 2*sd_val),
#   #               width = 0.2,position = position_dodge(0.2)) +
#   facet_grid(variable ~ family, scales = "free_y")
# 
# ggplot(summary_res, 
#        aes(x= factor(method), y=log(agg_val), 
#            fill = method)) + 
#   geom_boxplot(
#     #position = position_dodge(0.2)
#     ) + 
#   # geom_errorbar(aes(ymin = agg_val, ymax = agg_val + 2*sd_val),
#   #               width = 0.2,position = position_dodge(0.2)) +
#   facet_wrap(family ~ variable, scales = "free", nrow=1) + theme_bw() + xlab("") + ylab("log. RMSE") + 
#   coord_cartesian(ylim=c(-5,20))
# 
# library(knitr)
# library(kableExtra)
# 
# sres <- summary_res
# sres$variable[sres$variable!="prediction"] <- "weights"
# 
# sres %>% group_by(family, variable, method) %>% 
#   summarise("Median" = paste0(signif(median(agg_val, na.rm = T),3), 
#                               " (", signif(sd(agg_val, na.rm = T), 3),")")) %>% 
#   kable(format = 'latex', booktabs = TRUE) 
# 
# sres %>% group_by(family, variable, method) %>% 
#   summarise("Median" = paste0(signif(median(agg_val, na.rm = T),3), 
#                               " (", signif(quantile(agg_val, probs = c(0.9), na.rm = T), 3),")")) %>% 
#   kable(format = 'latex', booktabs = TRUE) 
# 
# 
# sres %>% group_by(method, family, variable) %>% filter(n==300, family!="lognormal") %>% 
#   summarise("Median" = paste0(signif(median(agg_val, na.rm = T),3), 
#                               " (", signif(mad(agg_val, na.rm = T), 3),")")) %>% 
#   .$Median %>% paste(., collapse = " & ")
# 
# sres %>% group_by(method, family, variable) %>% filter(n==2500, family!="lognormal") %>% 
#   summarise("Median" = paste0(signif(median(agg_val, na.rm = T),3), 
#                               " (", signif(mad(agg_val, na.rm = T), 3),")")) %>% 
#   .$Median %>% paste(., collapse = " & ")
# 
# 
# sres %>% group_by(method, family, variable) %>% filter(n==300, family!="lognormal") %>% 
#   summarise("CP" = sum(mean(agg_val) > 5*median(agg_val))) %>% as.data.frame()
# 
# sres %>% group_by(method, family, variable) %>% filter(n==2500, family!="lognormal") %>% 
#   summarise("CP" = sum(mean(agg_val) > 5*median(agg_val))) %>% as.data.frame()
