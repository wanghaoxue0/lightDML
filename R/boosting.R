# code for double machine learning with boosting tree
# Haoxue Wang 6 Feb. 2023
# hw613@cam.ac.uk


boosting <- function(data, y, x, d, z,score=c("ATE, LATE"), random_state=2023, verbose=FALSE) {
#split the sample into two parts
library(caret)
set.seed(random_state)
trainindex=createDataPartition(d,p=0.5,list=F)
data1=data[trainindex,]
data2=data[-trainindex,]
y1=y[trainindex]
y2=y[-trainindex]
d1=d[trainindex]
d2=d[-trainindex]

if(score=="ATE"){
  formula.1 <- as.formula(paste("y1~",x))
  formula.2 <- as.formula(paste("d1~",x))

  formula.3 <- as.formula(paste("y2~",x))
  formula.4 <- as.formula(paste("d2~",x))
}else if(score=="LATE"){
  formula.1 <- as.formula(paste("y1~",x))
  formula.2 <- as.formula(paste("d1~",x))
  formula.3 <- as.formula(paste("z1~",x))

  formula.4 <- as.formula(paste("y2~",x))
  formula.5 <- as.formula(paste("d2~",x))
  formula.6 <- as.formula(paste("z2~",x))
} else{
  cat("please specify the right score")
}

#########################################################
# Method 3: boosting
library(gbm)
set.seed(random_state)
#step 1
boost.aq=gbm(formula.1,data=data1,distribution="gaussian",n.trees=100,interaction.depth=4,shrinkage=0.1)
summary(boost.aq)
yhat.aq3=predict(boost.aq,newdata=data2,n.trees=100)
ylhat31=y2-yhat.aq3
#step 2
boost.d=gbm(formula.2,data=data1,distribution="gaussian",n.trees=100,interaction.depth=4,shrinkage=0.1)
summary(boost.d)
yhat.d3=predict(boost.d,newdata=data2,n.trees=100)
vhat31=d2-yhat.d3
#step 3
boost.aq=gbm(formula.3,data=data2,distribution="gaussian",n.trees=100,interaction.depth=4,shrinkage=0.1)
summary(boost.aq)
yhat.aq3=predict(boost.aq,newdata=data1,n.trees=100)
ylhat32=y1-yhat.aq3
#step 4
boost.d=gbm(formula.4,data=data2,distribution="gaussian",n.trees=100,interaction.depth=4,shrinkage=0.1)
summary(boost.d)
yhat.d3=predict(boost.d,newdata=data1,n.trees=100)
vhat32=d1-yhat.d3
#step5: reg ylhat vhat
lm.fit1=lm(ylhat31~vhat31)
summary(lm.fit1)

lm.fit2=lm(ylhat32~vhat32)
summary(lm.fit2)

# DML2: combine and reg
dim1=length(ylhat31)
dim2=length(ylhat32)
dim=dim1+dim2

dim3=dim1+1
yhat=rep(NA,dim)
yhat[1:dim1]=ylhat31
yhat[dim3:dim]=ylhat32

vhat=rep(NA,dim)
vhat[1:dim1]=vhat31
vhat[dim3:dim]=vhat32

lm.all=lm(yhat~vhat)
# compute robust standard error

#install.packages("lmtest")
#install.packages("sandwich")
library(lmtest)
library(sandwich)
est1 = coeftest(lm.fit1, vcov = vcovHC, type = "HC0")
est2 = coeftest(lm.fit2, vcov = vcovHC, type = "HC0")

b1 = est1[2,1]
b2 = est2[2,1]
be = (b1+b2)/2

se1 = est1[2,2]
se2 = est2[2,2]
sig2 =(se1^2+se2^2 +(b1-be)^2+(b2-be)^2)/2
se =sqrt(sig2)
t =be/se



  # combined reg
  estall = coeftest(lm.all, vcov = vcovHC, type = "HC0")
  beAll=estall[2,1]
  seAll=estall[2,2]
  tAll=beAll/seAll

if(verbose==TRUE){
  # ouput the estimation results
  cat("-----------------------------------------------------------","\n")
  cat("Double machine learning 1 (boosting, 2-folds):","\n")
  cat("Estimate, s.e., t-statistic, p.value, 95%lower, 95%upper","\n")
  print(cbind(theta=be,se.robust=se,t.value=t,pvalue=round(2*(1-pnorm(abs(be/se))),5),
              be-1.96*se,be+1.96*se),digits=4)

  cat("t-statistic critial values: 90%=1.65, 95%=1.96, 99%=2.58","\n")
  cat("-----------------------------------------------------------","\n")




  cat("-----------------------------------------------------------","\n")
  cat("Double machine learning 2 (boosting, 2-folds):","\n")
  cat("Estimate, s.e., t-statistic, p.value, 95%lower, 95%upper","\n")
  print(cbind(theta=beAll,se.robust=seAll,t.value=tAll,pvalue=round(2*(1-pnorm(abs(beAll/seAll))),5),
              beAll-1.96*seAll,beAll+1.96*seAll),digits=4)

  cat("t-statistic critial values: 90%=1.65, 95%=1.96, 99%=2.58","\n")
  cat("-----------------------------------------------------------","\n")

}

return(list(theta1=be,se.robust1=se, theta2=beAll,se.robust2=seAll))
}
