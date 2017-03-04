library(data.table)
library(psych)
library(mvtnorm)
library(caret)
source('selectThreshold.R')

# read data set
X <- fread(input = "creditcard.csv", sep = ",", header = T, showProgress = T)
X <- data.frame(X)
colnames(X)
mA <- sum(as.numeric(X$Class))/nrow(X)
sprintf('Percentage of fraudulent transactions in the data set %f', mA*100)
# split the data
rownames(X) <- 1:nrow(X)
Xgood <- X[X$Class == 0,]
Xanom <- X[X$Class == 1,]

# exploratory analysis
mugood <- apply(Xgood[sample(rownames(Xgood), size = as.integer(mA*nrow(X)), replace = F), -c(1, 30, 31)], 2, mean)
muanom <- apply(Xanom[, -c(1, 30, 31)], 2, mean)
plot(muanom, col = "blue", xlab = "Features", ylab = "Mean")
lines(muanom, col = "blue", lwd = 2)
points(mugood, col = "black")
lines(mugood, col = "black", lwd = 2)
legend("topright", legend = c("Good", "Anomalous"), lty = c(1,1), col = c("black", "blue"), lwd = c(2,2))

# drop features that are trivial
todrop <- paste("V", c(8, 13, 15, 19:28), sep = "")
Xgood[, todrop] <- NULL
Xanom[, todrop] <- NULL
# transform the features into more gaussian like
#Xgood[,-c(1,17,18)] <- log(Xgood[,-c(1,17,18)] + 200)
#Xanom[,-c(1,17,18)] <- log(Xanom[,-c(1,17,18)] + 200)
# plot the histogram of features
multi.hist(Xgood[,c(2,3,5,9)], density = T)

# split the data 60/20/20 into train/cv/test set with 0/50/50 anomalous examples 
g <- nrow(Xgood); a <- nrow(Xanom)
rg <- sample(1:g, g, replace = F)
ra <- sample(1:a, a, replace = F)
# training set
Xtrain <- data.matrix(Xgood[rg[1:(g*0.6)],-c(1,17)])
# cross validation set
Xcv <- data.matrix(rbind(Xgood[rg[(g*0.6+1):(g*0.8)],-c(1,17)], Xanom[ra[1:(a*0.5)],-c(1,17)]))
# test set
Xtest <- data.matrix(rbind(Xgood[rg[(g*0.8+1):g],-c(1,17)], Xanom[ra[(a*0.5 + 1):a],-c(1,17)]))
# shuffle the data
Xcv <- Xcv[sample(nrow(Xcv), nrow(Xcv), replace = F),]
Xtest <- Xcv[sample(nrow(Xtest), nrow(Xtest), replace = F),]
# cleanup variables that are not required
rm(list = c("a", "g", "ra", "rg", "todrop"))

# esimate gaussian
mu <- apply(Xtrain[,-16], 2, mean)       # mean of all features
sigma <- cov(Xtrain[,-16])               # covariance matrix of all features
#sigma <- apply(Xtrain[,-16], 2, var)    
#sigma <- diag(sigma)                      # assuming all the features are uncorrelated

# find the pdf on cross validation set
pcv <- dmvnorm(Xcv[,-16], mu, sigma)
# find the best threshold for maximum F1 score
epsilon <- selectThreshold(Xcv[,16], pcv)
sprintf('The threshold value is found to be epsilon = %e', epsilon)

# evaluate the model on test set
ptest <- dmvnorm(Xtest[,-16], mu, sigma)
prediction <- as.numeric(ptest < epsilon)
confusionMatrix(table(Xtest[,16], prediction), positive = levels(factor(Xtest[,16]))[2])

