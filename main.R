library(dplyr)
library(mvtnorm)
X <- read.csv(file = "creditcard.csv", header = T)
# split the data 60/20/20 into train/cv/test set with 0/50/50 anomalous examples 
Xgood <- X[X$Class == 0, -c(1,30)]
Xbad <- X[X$Class == 1, -c(1,30)]

mugood <- apply(Xgood[,-29], 2, mean)
mubad <- apply(Xbad[,-29], 2, mean)

mugood <- exp(mugood)
mubad <- exp(mubad)
mugood <- mugood/sum(mugood)
mubad <- mubad/sum(mubad)
# plot the mean values of all features
plot(mubad, col = "blue")
lines(mubad, col = "blue")
points(mugood)
lines(mugood)
# only keep interesting features
# todrop <- c(5, 6, 8, 9, 13, 15, 16, 18:28)
Xgood <- cbind(Xgood$V2^2, Xgood$V4^2, Xgood$V11^2, Xgood$V2*Xgood$V4, Xgood$V2*Xgood$V11, Xgood$V4*Xgood$V11, Xgood$Class)
Xbad <- cbind(Xbad$V2^2, Xbad$V4^2, Xbad$V11^2, Xbad$V2*Xbad$V4, Xbad$V2*Xbad$V11, Xbad$V4*Xbad$V11, Xbad$Class)
todrop <- c(paste("f", 1:6, sep = ""), "Class")
colnames(Xgood) <- todrop
colnames(Xbad) <- todrop
# distribute good and bad data to find best epsilon
g <- nrow(Xgood); b <- nrow(Xbad)
rg <- sample(1:g, g, replace = F)
rb <- sample(1:b, b, replace = F)

Xtrain <- data.matrix(Xgood[rg[1:(g*0.6)],])
Xcv <- data.matrix(rbind(Xgood[rg[(g*0.6 + 1):(g*0.8)],], Xbad[rb[1:(b*0.5)],]))
Xtest <- data.matrix(rbind(Xgood[rg[(g*0.8 + 1):g],], Xbad[rb[(b*0.5 + 1):b],]))
rm(list = c("b", "g", "rb", "rg", "todrop"))

# estimate guassian on training set
mu <- apply(Xtrain[,-7], 2, mean)       # mean of all features
# sigma <- cov(Xtrain[,-11])               # covariance matrix of all features
sigma <- apply(Xtrain[,-7], 2, var)
sigma <- diag(sigma)

# find probabilities on cross validation
pcv <- dmvnorm(Xcv[,-7], mu, sigma)
#pcv1 <- mvGaussian(Xcv[,-29], mu, sigma)

epsilon <- selectThreshold(Xcv[,7], pcv)

ptest <- dmvnorm(Xtest[,-7], mu, sigma)
ytest <- ptest < epsilon

# model leading to too many zero probabilities for anomalous examples
# need to manipulate features to fix this issue
