selectThreshold_1 <- function(ycv, pcv)
{
library(caret)
# initiate variales
pcv <- pcv/sum(pcv); bestEpsilon <- 0; epsilon <- 0; bestF1 <- 0;
stepsize = (max(pcv) - min(pcv)) / 100

# loop through a range of epsilon values
for (epsilon in seq(min(pcv)+stepsize, max(pcv), stepsize)) {
  pred <- as.numeric(pcv < epsilon)
  errMet <- confusionMatrix(table(ycv, pred), positive = levels(factor(ycv))[2])
  # find best epsilon
  if (errMet$byClass["F1"] > bestF1) {
    bestF1 <- errMet$byClass["F1"]
    bestEpsilon <- epsilon
    message(sprintf('best f1: %f, best epsilon: %f\n', bestF1, bestEpsilon))
  } 
}
return(bestEpsilon)
}