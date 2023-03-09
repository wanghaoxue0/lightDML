# code for double machine learning with bagging trees
# Haoxue Wang 6 Feb. 2023
# hw613@cam.ac.uk

bagging <- function(data, y, x, d, z,score=c("ATE, LATE"), random_state=2023, verbose=FALSE) {

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
  formula.1 <- as.formula(paste("y~",x))
  formula.2 <- as.formula(paste("d~",x))

  formula.3 <- as.formula(paste("y~",x))
  formula.4 <- as.formula(paste("d~",x))
} else{
  cat("please specify the right score")
}



#########################################################
if(score=="ATE"){
# Method 1: bagging

mtry = ncol(data)-2
library(randomForest)
set.seed(random_state)
#step 1
bag.aq=randomForest(formula.1,data=data1,mtry=mtry,importance=TRUE)
summary(bag.aq)
yhat.aq = predict(bag.aq, newdata = data2)
ylhat1=y2-yhat.aq
#step 2
bag.d=randomForest(formula.2,data=data1,mtry=mtry,importance=TRUE)
summary(bag.d)
yhat.d = predict(bag.d, newdata = data2)
vhat1=d2-yhat.d
#step 3
bag.aq=randomForest(formula.3,data=data2,mtry=mtry,importance=TRUE)
summary(bag.aq)
yhat.aq = predict(bag.aq, newdata = data1)
ylhat2=y1-yhat.aq
#step 4
bag.d=randomForest(formula.4,data=data2,mtry=mtry,importance=TRUE)
summary(bag.d)
yhat.d = predict(bag.d, newdata = data1)
vhat2=d1-yhat.d
#step5: reg ylhat vhat
lm.fit1=lm(ylhat1~vhat1)
summary(lm.fit1)

lm.fit2=lm(ylhat2~vhat2)
summary(lm.fit2)

# DML2: combine and reg
dim1=length(ylhat1)
dim2=length(ylhat2)
dim=dim1+dim2

dim3=dim1+1
yhat=rep(NA,dim)
yhat[1:dim1]=ylhat1
yhat[dim3:dim]=ylhat2

vhat=rep(NA,dim)
vhat[1:dim1]=vhat1
vhat[dim3:dim]=vhat2

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

}else if(score=="LATE"){

  mtry = ncol(data)-2
  library(randomForest)
  set.seed(random_state)
  #step 1
  bag.aq1=randomForest(formula.1,data=data1[which(data1$z==1),],mtry=mtry,importance=TRUE)
  summary(bag.aq1)
  yhat.aq1 = predict(bag.aq1, newdata = data2[which(data2$z==1),])
  ylhat11=y2[which(data2$z==1)]-yhat.aq1

  bag.aq0=randomForest(formula.2,data=data1[which(data1$z==0),],mtry=mtry,importance=TRUE)
  summary(bag.aq0)
  yhat.aq0 = predict(bag.aq0, newdata = data2[which(data2$z==0),])
  ylhat10=y2[which(data2$z==0)]-yhat.aq0
  ylhat1=mean(ylhat11)-mean(ylhat10)

  #step 2
  bag.d1=randomForest(formula.3,data=data1[which(data1$z==1),],mtry=mtry,importance=TRUE)
  summary(bag.d1)
  yhat.d1 = predict(bag.d, newdata = data2[which(data2$z==1),])
  vhat11=d2[which(data2$z==1)]-yhat.d1

  bag.d0=randomForest(formula.4,data=data1[which(data1$z==0),],mtry=mtry,importance=TRUE)
  summary(bag.d)
  yhat.d0 = predict(bag.d0, newdata = data2[which(data2$z==0),])
  vhat10=d2[which(data2$z==0)]-yhat.d0
  vlhat1=mean(vhat11)-mean(vhat10)

  theta1 = ylhat1/vlhat1

  #step3
  bag.aq1=randomForest(formula.1,data=data2[which(data2$z==1),],mtry=mtry,importance=TRUE)
  summary(bag.aq1)
  yhat.aq1 = predict(bag.aq1, newdata = data1[which(data1$z==1),])
  ylhat11=y2[which(data1$z==1)]-yhat.aq1

  bag.aq0=randomForest(formula.2,data=data2[which(data2$z==0),],mtry=mtry,importance=TRUE)
  summary(bag.aq0)
  yhat.aq0 = predict(bag.aq0, newdata = data1[which(data1$z==0),])
  ylhat10=y2[which(data1$z==0)]-yhat.aq0
  ylhat1=mean(ylhat11)-mean(ylhat10)

  #step 4
  bag.d1=randomForest(formula.3,data=data2[which(data2$z==1),],mtry=mtry,importance=TRUE)
  summary(bag.d1)
  yhat.d1 = predict(bag.d, newdata = data1[which(data1$z==1),])
  vhat11=d2[which(data1$z==1)]-yhat.d1

  bag.d0=randomForest(formula.4,data=data2[which(data2$z==0),],mtry=mtry,importance=TRUE)
  summary(bag.d)
  yhat.d0 = predict(bag.d0, newdata = data1[which(data1$z==0),])
  vhat10=d2[which(data1$z==0)]-yhat.d0
  vlhat1=mean(vhat11)-mean(vhat10)
  vlhat1=mean(vhat11)-mean(vhat10)

  theta2 = ylhat1/vlhat1
}


if(verbose==TRUE){
# ouput the estimation results
cat("-----------------------------------------------------------","\n")
cat("Double machine learning 1 (bagging, 2-folds):","\n")
cat("Estimate, s.e., t-statistic, p.value, 95%lower, 95%upper","\n")
print(cbind(theta=be,se.robust=se,t.value=t,pvalue=round(2*(1-pnorm(abs(be/se))),5),
            be-1.96*se, be+1.96*se),digits=4)

cat("t-statistic critial values: 90%=1.65, 95%=1.96, 99%=2.58","\n")
cat("-----------------------------------------------------------","\n")




cat("-----------------------------------------------------------","\n")
cat("Double machine learning 2 (bagging, 2-folds):","\n")
cat("Estimate, s.e., t-statistic, p.value, 95%lower, 95%upper","\n")
print(cbind(theta=beAll,se.robust=seAll,t.value=tAll,pvalue=round(2*(1-pnorm(abs(beAll/seAll))),5),
              beAll-1.96*seAll,beAll+1.96*seAll),digits=4)

cat("t-statistic critial values: 90%=1.65, 95%=1.96, 99%=2.58","\n")
cat("-----------------------------------------------------------","\n")
}
return(list(theta1=be,se.robust1=se, theta2=beAll,se.robust2=seAll))
}
