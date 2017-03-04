selectThreshold <- function(ycv, pcv)
{
# initiate variales
bestEpsilon <- 0; epsilon <- 0; bestF1 <- 0; F1 <- 0; 
stepsize = (max(pcv) - min(pcv)) / 10000000

# loop through a range of epsilon values
for (epsilon in seq(min(pcv)+stepsize, max(pcv), stepsize)) {
  pred <- as.numeric(pcv < epsilon)
  # calculate precision and recall
  tp <- sum((pred == 1) & (ycv == 1))   # true positives
  fp <- sum((pred == 1) & (ycv == 0))   # false positives
  fn <- sum((pred == 0) & (ycv == 1))   # false negatives

  #if (tp + fp == 0) prec <- 0 else 
  prec <- tp/(tp + fp); rec <- tp/(tp + fn);
  
  # calculate F score
  #if (prec + rec == 0) F1 <- 0 else 
  F1 <- (2 * prec * rec)/(prec + rec)
  # find best epsilon
  if (F1 > bestF1) {
    bestF1 <- F1
    bestEpsilon <- epsilon
    #message(sprintf('best f1: %e, best epsilon: %e\n', bestF1, bestEpsilon))
    loopct <- 1
  } 
  loopct = loopct+1
  if (loopct > 10000) break
}
return(bestEpsilon)
}