char_to_fun <- function(char) eval(parse(text = paste0(char, '()')))

center_nl <- function(fun){
  deepregression:::orthog_structured(fun, matrix(rep(1,length(fun)),
                                                 ncol=1))
}
