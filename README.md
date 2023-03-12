# Light Double Machine Learning

### Haoxue Wang

University of Cambridge

hw613@cam.ac.uk

```
library(devtools)
install_github("wanghaoxue0/lightDML")
library(randomForest)
library(gbm)
library(lightDML)
model <- fit(data, y, x, d, z=NA, ml=c("bagging","boosting","random forest", "neural network"), score=c("ATE, LATE"), random_state=2023, verbose=FALSE)
```
```
Package: lightDML
Type: Package
Title: A light version of double machine learning with instrumental variable
Version: 0.1.0
Author: Haoxue Wang
Maintainer: Haoxue Wang <hw613@cam.ac.uk>
Description: A light version of double machine learning with instrumental variables,
focusing on estimating ATE/LATE with tree-based algorithms. This package is more
for users who have little knowledge on double machine learning methods.
License: MIT
Encoding: UTF-8
LazyData: true
```

You can conduct the simulation in the following way
```
source("bagging.R")
source("boosting.R")
source("rf.R")
source("nn.R")
source("fit.R")


# generate the data
rep= 25   #simulation replications

est1 <- matrix(rep(NA, 4 * rep), ncol = 4)  # regression on seperate residuals
est2 <- matrix(rep(NA, 4 * rep), ncol = 4)  # combine the residuals
colnames(est1) <-c("bagging","boosting", "randomforest", "NN")
colnames(est2) <-c("bagging","boosting", "randomforest", "NN")



# simlation for mapping with linear term
set.seed(NULL)
for(i in 1:rep){
  #generate simulated data
  n=100#sample size
  # n=500
  # n=1000
  # n=5000
  # n=10000
  u=rnorm(n)
  v=rnorm(n)
  x1=runif(n)
  x2=runif(n)


  mx=(x1+x2)/2
  fx=-5+6*x1+7*x2

  d=rbinom(n,1,prob=mx)

  y <- 1*d+fx+u

  x=cbind.data.frame(x1,x2)
  data=data.frame(y,x,d)

  x="x1+x2"
  seed <- 2023

  # 2-fold validation
  fit1 <- fit(data=data,y,x,d, score="ATE", ml="bagging", random_state =seed)
  est1[i,1] <- fit1$theta1
  est2[i,1] <- fit1$theta2

  fit2 <- fit(data=data,y,x,d, score="ATE", ml="boosting", random_state =seed)
  est1[i,2] <- fit2$theta1
  est2[i,2] <- fit2$theta2

  fit3 <- fit(data=data,y,x,d, score="ATE", ml="random forest", random_state =seed)
  est1[i,3] <- fit3$theta1
  est2[i,3] <- fit3$theta2

  fit4 <- fit(data=data,y,x,d, score="ATE", ml="neural network", random_state =seed)
  est1[i,4] <- fit4$theta1
  est2[i,4] <- fit4$theta2

}

mi1=colMeans(est1)
sd1=cbind(sd(est1[,1]),sd(est1[,2]),sd(est1[,3]),sd(est1[,4]))
print(list(mi1,sd1))

mi2=colMeans(est2)
sd2=cbind(sd(est2[,1]),sd(est2[,2]),sd(est2[,3]),sd(est2[,4]))
print(list(mi2,sd2))

```
