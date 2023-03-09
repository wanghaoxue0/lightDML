# code for double machine learning with random forest
# Haoxue Wang 6 Feb. 2023
# hw613@cam.ac.uk

rf <- function(data, y, x, d, z,score=c("ATE, LATE"), random_state=2023, verbose=FALSE) {
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
# Method 2: random forecast
library(randomForest)
set.seed(random_state)
#step 1
rf.aq2=randomForest(formula.1,data=data1,ntree=25,importance=TRUE)
summary(rf.aq2)
yhat.aq2 = predict(rf.aq2, newdata = data2)
ylhat21=y2-yhat.aq2
#step 2
rf.d2=randomForest(formula.2,data=data1,ntree=25,importance=TRUE)
summary(rf.d2)
yhat.d2 = predict(rf.d2, newdata = data2)
vhat21 = d2-yhat.d2
#step 3
rf.aq2=randomForest(formula.3,data=data2,ntree=25,importance=TRUE)
summary(rf.aq2)
yhat.aq2 = predict(rf.aq2, newdata = data1)
ylhat22 = y1-yhat.aq2
#step 4
rf.d2=randomForest(formula.4,data=data2,ntree=25,importance=TRUE)
summary(rf.d2)
yhat.d2 = predict(rf.d2, newdata = data1)
vhat22 = d1-yhat.d2

#step5: reg ylhat vhat
lm.fit1=lm(ylhat21~vhat21)
summary(lm.fit1)

lm.fit2=lm(ylhat22~vhat22)
summary(lm.fit2)

# DML2: combine and reg
dim1=length(ylhat21)
dim2=length(ylhat22)
dim=dim1+dim2

dim3=dim1+1
yhat=rep(NA,dim)
yhat[1:dim1]=ylhat21
yhat[dim3:dim]=ylhat22

vhat=rep(NA,dim)
vhat[1:dim1]=vhat21
vhat[dim3:dim]=vhat22

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
cat("Double machine learning 1 (random forecast , 2-folds):","\n")
cat("Estimate, s.e., t-statistic, p.value, 95%lower, 95%upper","\n")
print(cbind(theta=be,se.robust=se,t.value=t,pvalue=round(2*(1-pnorm(abs(be/se))),5),
              be-1.96*se,be+1.96*se),digits=4)

cat("t-statistic critial values: 90%=1.65, 95%=1.96, 99%=2.58","\n")
cat("-----------------------------------------------------------","\n")




cat("-----------------------------------------------------------","\n")
cat("Double machine learning 2 (random forecast, 2-folds):","\n")
cat("Estimate, s.e., t-statistic, p.value, 95%lower, 95%upper","\n")
print(cbind(theta=beAll,se.robust=seAll,t.value=tAll,pvalue=round(2*(1-pnorm(abs(beAll/seAll))),5),
              beAll-1.96*seAll,beAll+1.96*seAll),digits=4)

cat("t-statistic critial values: 90%=1.65, 95%=1.96, 99%=2.58","\n")
cat("-----------------------------------------------------------","\n")
}
return(list(theta1=be,se.robust1=se, theta2=beAll,se.robust2=seAll))
}






