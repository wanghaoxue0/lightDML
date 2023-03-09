# simulation code for LATE
# Haoxue Wang 6 Feb. 2023
# hw613@cam.ac.uk

# source the code we need
source("bagging.R")
source("boosting.R")
source("rf.R")
source("nn.R")
source("fit.R")


# generate the data
rep= 5   #simulation replications

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

  px=x1
  mx=(x1-x2)/2
  mux=-5+6*x1+7*x2

  z=rbinom(n,1, prob=px)
  d=1*(z+mx+v>0)

  y <- 1*d+mux+u
  x=cbind.data.frame(x1,x2)
  data=data.frame(y,x,d,z)
  x="x1+x2"
  seed <- 2023

  fit1 <- fit(data=data,y,x,d,z, score="LATE", ml="bagging", random_state =seed)
  est1[i,1] <- fit1[[1]]
  est2[i,1] <- fit1[[2]]

  fit2 <- fit(data=data,y,x,d,z, score="LATE", ml="boosting", random_state =seed)
  est1[i,2] <- fit2[[1]]
  est2[i,2] <- fit2[[2]]

  fit3 <- fit(data=data,y,x,d,z, score="LATE", ml="random forest", random_state =seed)
  est1[i,3] <- fit3[[1]]
  est2[i,3] <- fit3[[2]]

  fit4 <- fit(data=data,y,x,d,z, score="LATE", ml="neural network", random_state =seed)
  est1[i,4] <- fit4[[1]]
  est2[i,4] <- fit4[[2]]
}
# simulation for mapping with an interaction term
colMeans(est1)
colMeans(est2)
rep=1

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

  px=rbinom(n,1,prob=0.5)
  dx = x1-x2+x3
  mx=1/(1+exp(-dx))
  mux=-5+6*x1+7*x2+8*x3

  z=px
  d=1*(z+mx+v>0)
  y <- 1*d+mux+u
  x=cbind.data.frame(x1,x2,x3)
  data=data.frame(y,x,d,z)

  x="x1+x2+x3"
  seed <- 2023

  fit1 <- fit(data=data,y,x,d,z, score="LATE", ml="bagging", random_state =seed)
  est1[i,1] <- fit1[[1]]
  est2[i,1] <- fit1[[2]]

  fit2 <- fit(data=data,y,x,d,z, score="LATE", ml="boosting", random_state =seed)
  est1[i,2] <- fit2[[1]]
  est2[i,2] <- fit2[[2]]

  fit3 <- fit(data=data,y,x,d,z, score="LATE", ml="random forest", random_state =seed)
  est1[i,3] <- fit3[[1]]
  est2[i,3] <- fit3[[2]]

  fit4 <- fit(data=data,y,x,d,z, score="LATE", ml="neural network", random_state =seed)
  est1[i,4] <- fit4[[1]]
  est2[i,4] <- fit4[[2]]

}
round(colMeans(est1),4)
colMeans(est2)


# simulation for mapping with the square term

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

  px=x1
  dx = x1+x2+x3+x4
  mx=1/(1+exp(-dx))
  mux=-5+6*x1+7*x2+8*x3+9*x4

  z=rbinom(n,1,prob=0.5)
  d=1*(z+mx+v>0)
  y <- 1*d+mux+u
  x=cbind.data.frame(x1,x2,x3,x4)
  data=data.frame(y,x,z,d)

  x="x1+x2+x3+x4"
  seed <- 2023

  fit1 <- fit(data=data,y,x,d,z, score="LATE", ml="bagging", random_state =seed)
  est1[i,1] <- fit1[[1]]
  est2[i,1] <- fit1[[2]]

  fit2 <- fit(data=data,y,x,d,z, score="LATE", ml="boosting", random_state =seed)
  est1[i,2] <- fit2[[1]]
  est2[i,2] <- fit2[[2]]

  fit3 <- fit(data=data,y,x,d,z, score="LATE", ml="random forest", random_state =seed)
  est1[i,3] <- fit3[[1]]
  est2[i,3] <- fit3[[2]]

  fit4 <- fit(data=data,y,x,d,z, score="LATE", ml="neural network", random_state =seed)
  est1[i,4] <- fit4[[1]]
  est2[i,4] <- fit4[[2]]

}

colMeans(est1)
colMeans(est2)
set.seed(2023)
# simulation for mapping with interaction and square term
for(i in 1:rep){
  #generate simulated data
  n=1000#sample size
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
  mux=-5+6*x1+7*x2+8*x3+9*x4+10*x5

  z=rbinom(n,1,prob=0.5)
  d=1*(z+mx+v>0)
  y <- 1*d+mux+u
  x=cbind.data.frame(x1,x2,x3,x4,x5)
  data=data.frame(y,x,d,z)

  x="x1+x2+x3+x4+x5"
  seed <- 2023

  fit1 <- fit(data=data,y,x,d,z, score="LATE", ml="bagging", random_state =seed)
  est1[i,1] <- fit1[[1]]
  est2[i,1] <- fit1[[2]]

  fit2 <- fit(data=data,y,x,d,z, score="LATE", ml="boosting", random_state =seed)
  est1[i,2] <- fit2[[1]]
  est2[i,2] <- fit2[[2]]

  fit3 <- fit(data=data,y,x,d,z, score="LATE", ml="random forest", random_state =seed)
  est1[i,3] <- fit3[[1]]
  est2[i,3] <- fit3[[2]]

  fit4 <- fit(data=data,y,x,d,z, score="LATE", ml="neural network", random_state =seed)
  est1[i,4] <- fit4[[1]]
  est2[i,4] <- fit4[[2]]

}
colMeans(est1)
colMeans(est2)
