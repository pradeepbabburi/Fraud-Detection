mvGaussian <- function(x, mu, sigma) {
  k <- length(mu)
  x <- x - mu
  x <- data.matrix(x)
  C <- (2 * pi)^(-k / 2)*det(sigma)^(-0.5)
  sigma.inv <- solve(sigma)
  tmp <- NULL
  #for (i in 1:nrow(X)) {
    tmp <- exp(-0.5 * x %*% sigma.inv %*% t(x))
  #}
  return(C*tmp)
}