mvGaussian <- function(X, mu, sigma) {
  k = length(mu)
  X = X - mu
  C = (2 * pi)^(-k / 2)*det(sigma)^(-0.5)
  sigma.inv <- solve(sigma)
  tmp <- NULL
  for (i in 1:nrow(X)) {
    tmp <- c(tmp, exp(-0.5 * t(X[i,]) %*% sigma.inv %*% matrix(X[i,])))
  }
  return(C*tmp)
}