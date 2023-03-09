# simulation code for ATE
# Haoxue Wang 6 Feb. 2023
# hw613@cam.ac.uk

# source the code we need
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




# simulation for mapping with an interaction term

for(i in 1:rep){
  #generate simulated data
  n=5000#sample size
  # n=500
  # n=1000
  # n=5000
  # n=10000
  u=rnorm(n)
  v=rnorm(n)
  x1=rnorm(n)
  x2=rnorm(n)
  x3=x1*x2

  dx = x1+x2+x3
  mx=1/(1+exp(-dx))
  fx=-5+6*x1+7*x2+8*x3

  d=rbinom(n,1,prob=mx)
  y <- 1*d+fx+u
  x=cbind.data.frame(x1,x2,x3)
  data=data.frame(y,x,d)

  x="x1+x2+x3"
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


# simulation for mapping with the square term
set.seed(NULL)
for(i in 1:rep){
  #generate simulated data
  n=5000#sample size
  # n=500
  # n=1000
  # n=5000
  # n=10000
  u=rnorm(n)
  v=rnorm(n)
  x1=rnorm(n)
  x2=rnorm(n)
  x3=x1*x1
  x4=x2*x2

  dx = x1+x2+x3+x4
  mx=1/(1+exp(-dx))
  fx=-5+6*x1+7*x2+8*x3+9*x4

  d=rbinom(n,1,prob=mx)
  y <- 1*d+fx+u
  x=cbind.data.frame(x1,x2,x3,x4)
  data=data.frame(y,x,d)

  x="x1+x2+x3+x4"
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

set.seed(NULL)
# simulation for mapping with interaction and square term
for(i in 1:rep){
  #generate simulated data
  n=5000#sample size
  # n=500
  # n=1000
  # n=5000
  # n=10000
  u=rnorm(n)
  v=rnorm(n)
  x1=rnorm(n)
  x2=rnorm(n)
  x3=x1*x1
  x4=x2*x2
  x5=x1*x2

  dx = x1+x2+x3+x4
  mx=1/(1+exp(-dx))
  fx=-5+6*x1+7*x2+8*x3+9*x4+10*x5

  d=rbinom(n,1,prob=mx)
  y <- 1*d+fx+u
  x=cbind.data.frame(x1,x2,x3,x4,x5)
  data=data.frame(y,x,d)

  x="x1+x2+x3+x4+x5"
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

