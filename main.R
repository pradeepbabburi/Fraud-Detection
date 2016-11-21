library(dplyr)
library(mvtnorm)
X <- read.csv(file = "creditcard.csv", header = T)
# split the data 60/20/20 into train/cv/test set with 0/50/50 anomalous examples 
Xgood <- X[X$Class == 0, -c(1,30)]
Xbad <- X[X$Class == 1, -c(1,30)]

g <- nrow(Xgood); b <- nrow(Xbad)
rg <- sample(1:g, g, replace = F)
rb <- sample(1:b, b, replace = F)
# split good data
Xtrain <- data.matrix(Xgood[rg[1:(g*0.6)],])
Xcv <- data.matrix(rbind(Xgood[rg[(g*0.6 + 1):(g*0.8)],], Xbad[rb[1:(b*0.5)],]))
Xtest <- data.matrix(rbind(Xgood[rg[(g*0.8 + 1):g],], Xbad[rb[(b*0.5 + 1):b],]))
rm(list = c("b", "g", "rb", "rg"))

# estimate guassian on training set
mu <- apply(Xtrain[,-29], 2, mean)       # mean of all features
sigma <- cov(Xtrain[,-29])               # covariance matrix of all features

pcv <- dmvnorm(Xcv[,-29], mu, sigma)
#pcv1 <- mvGaussian(Xcv[,-29], mu, sigma)

epsilon <- selectThreshold(Xcv[,29], pcv)

ptest <- dmvnorm(Xtest[,-29], mu, sigma)
ytest <- ptest < epsilon

# model leading to too many zero probabilities for anomalous examples
# need to manipulate features to fix this issue
