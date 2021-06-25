dgp <- function(
  n, # number of observations
  p1, # number of linear features first parameter
  p2, # number of linear features second parameter
  pnl1, # number of non-linear features first parameter
  pnl2, # number of non-linear features second paramter
  overlapl, # number of linear features to overlap among parameters
  overlapnl, # number of non-linear features to overlap among parameters
  functions = list(function(x) cos(5*x),
                   function(x) tanh(3*x),
                   function(x) -x^3,
                   function(x) cos(x*3-2)*(-x*3),
                   function(x) exp(x*2) - 1,
                   function(x) x^2,
                   function(x) sin(x)*cos(x),
                   function(x) sqrt(abs(x)),
                   function(x) -x^5,
                   function(x) log(abs(x)^2)/100,
                   function(x) sin(10*x),
                   function(x) (sin(10*x)*0.1 + 1) * (x<0) + (-2*x + 1) * (x >=0),
                   function(x) -x * tanh(3*x) * sin(4*x)
  ),
  add_error1 = rnorm(n,0,0.000001), # additional error for first linear predictor
  add_error2 = rnorm(n,0,0.000001), # additional error for second linear preditor
  add_high_ia1 = FALSE,
  add_high_ia2 = FALSE,
  bias1 = 1,
  bias2 = -1,
  outcome_trafo,
  coefseq = function(){set.seed(42); runif(p1 - overlapl + p2, -3, 3)},
  divideBy = 1
)
{
  
  norg <- n
  n <- round(n * 5/4) # to add a test set
  
  X <- matrix(runif(n * (p1 + p2 + pnl1 + pnl2), -1, 1), nrow=n)
  colnames(X) <- paste0("V", 1:ncol(X))
  X <- scale(X, scale = F)
  
  coef_seq <- coefseq()
  
  # initialize indices and effects
  ind_lin1 <- ind_lin2 <- ind_nl1 <- ind_nl2 <- c()
  
  part1pred_l <- part2pred_l <- 
    part1pred_nl <- part2pred_nl <- 
    interact1 <- interact2 <- matrix(0, nrow=n)
  
  # define indices
  if(p1 > 0) ind_lin1 <- 1:p1
  if(p2 > 0) ind_lin2 <- 
    (p1 - overlapl + 1):(
      p1 - overlapl + p2)
  if(pnl1 > 0) ind_nl1 <- 
    (p1 + p2 - overlapl + 1):(
      p1 + p2 - overlapl + pnl1)
  if(pnl2 > 0) ind_nl2 <- 
    (p1 + p2 - overlapl + pnl1 - overlapnl + 1):(
      p1 + p2 - overlapl + pnl1 - overlapnl + pnl2)
  
  # linear terms
  if(p1 > 0) part1pred_l <- sapply(ind_lin1, function(j) coef_seq[j]*X[,j])
  if(p2 > 0) part2pred_l <- sapply(ind_lin2, function(j) coef_seq[j]*X[,j])
  
  # interactions
  if(add_high_ia1) interact1 <- matrix(apply(X[,ind_lin1], 1, prod), ncol=1)
  if(add_high_ia2) interact2 <- matrix(apply(X[,ind_lin2], 1, prod), ncol=1)
  
  if(pnl1 > 0) part1pred_nl <- sapply(ind_nl1, function(j) 
    center_nl(functions[[j-min(ind_nl1)+1]](X[,j])))
  if(pnl2 > 0) part2pred_nl <- sapply(ind_nl2, function(j) 
    center_nl(functions[[j-min(ind_nl1)+1]](X[,j])))
  
  # define true linear predictors
  true_linpred1 <- bias1 + 
    rowSums(part1pred_l) + 
    rowSums(part1pred_nl) + 
    interact1
  
  true_linpred2 <- bias2 + 
    rowSums(part2pred_l) + 
    rowSums(part2pred_nl) + 
    interact2 

  outcome <- outcome_trafo(n = n,
                           (true_linpred1 + add_error1)[,1]/divideBy, 
                           (true_linpred2 + add_error2)[,1]/divideBy)
  
  f1 <- "~ 1 +"
  if(p1 > 0) f1 <- paste(f1, paste(colnames(X)[ind_lin1],
                                   collapse = "+"), collapse = "+")
  if(pnl1 > 0) f1 <- paste(f1, " + s(", paste(colnames(X)[ind_nl1], 
                                              collapse = ") + s("),
                           ")")
  f2 <- "~ 1 +"
  if(p2 > 0) f2 <- paste(f2, paste(colnames(X)[ind_lin2],
                                   collapse = " + "), collapse = "+")
  if(pnl2 > 0) f2 <- paste(f2, " + s(", paste(colnames(X)[ind_nl2], 
                                              collapse = ") + s("),
                           ")")
  
  list_of_formulae <- c(f1,f2)
  
  return(
    list(y = outcome,
         X = as.data.frame(X),
         list_of_formulae = list_of_formulae,
         bias1 = bias1/divideBy,
         bias2 = bias2/divideBy,
         lincoef1 = coef_seq[ind_lin1]/divideBy,
         lincoef2 = coef_seq[ind_lin2]/divideBy,
         nlfun1 = part1pred_nl/divideBy,
         nlfun2 = part2pred_nl/divideBy,
         true_linpred1 = true_linpred1/divideBy,
         true_linpred2 = true_linpred2/divideBy,
         interaction1 = add_high_ia1/divideBy,
         interaction2 = add_high_ia2/divideBy,
         error1 = add_error1/divideBy,
         error2 = add_error2/divideBy,
         train = 1:norg,
         test = (norg+1):n,
         ind_lin1 = ind_lin1, 
         ind_lin2 = ind_lin2,
         ind_nl1 = ind_nl1, 
         ind_nl2 = ind_nl2
    )
  )
    
}