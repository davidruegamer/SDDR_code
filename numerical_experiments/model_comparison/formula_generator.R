formula_generator <- function(lof, this_family)
{
  
  # this_family <- family[["Family"]]
  
  bamlss_resp <- char_to_fun(this_family[["bamlss"]])$names
  bamlss_resp[1] <- "y"
  mboostlss_resp <- c("mu","sigma")
  gamlss_resp <- c("y", rep("", 5))
  replace_s_bbs <- function(form){ 
    gsub("s\\((V[0-9]*) \\)", "bbs\\(\\1, df\\=1, center\\=TRUE\\)",
         gsub("s\\( (V[0-9]*)\\)", "bbs\\(\\1, df\\=1, center\\=TRUE\\)",
              gsub("s\\((V[0-9]*)\\)", "bbs\\(\\1, df\\=1, center\\=TRUE\\)",
                   gsub("s\\( (V[0-9]*) \\)", 
                        "bbs\\(\\1, df\\=1, center\\=TRUE\\)", form))))
  }
  replace_nos_bols <- function(form) 
    gsub("\\+(V[0-9]*)",
         " \\+ bols\\(\\1, intercept \\= FALSE, df \\= 1\\)",
         gsub("\\+\\s","\\+",form))
  #)
  replace_one <- function(form) gsub("~ 1  \\+", "~ ", form)
  replace_s_pb <- function(form) gsub("s\\(", "pb\\(", form)
  name_list <- function(list,nam){names(list) <- nam;return(list)}
  
  bamlss_form <- lapply(1:length(lof), function(i) 
    as.formula(paste(bamlss_resp[i], lof[[i]])))
  if(this_family$Family=="logistic")
    bamlss_form <- name_list(c(bamlss_form, list(~1)),
                             c("mu","sigma","alpha"))
  
  ret <- list(
    deepregression = 
      lapply(lof, as.formula),
    bamlss = bamlss_form,
    gamlss = 
      c(lapply(1:length(lof), function(i) 
        as.formula(paste(gamlss_resp[i], replace_s_pb(lof[[i]])))),
        lapply(rep("~1", 4-length(lof)), as.formula)
        ),
    mboostlss = 
      name_list(lapply(1:length(lof), function(i) 
        as.formula(paste("y", replace_one(
          replace_nos_bols(replace_s_bbs(lof[[i]])))))),
                mboostlss_resp)
  )
  
  return(ret)
  
}
