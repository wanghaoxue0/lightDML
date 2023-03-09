# code for double machine learning with neural network
# Haoxue Wang 6 Feb. 2023
# hw613@cam.ac.uk


nn <- function(data, y, x, d, z,score=c("ATE, LATE"), random_state=2023, verbose=FALSE) {
#split the sample into two parts

set.seed(random_state)
library(caret)
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
ns <- ncol(data)-2
# method 4: nuetral networks
library(nnet)
set.seed(random_state)
#step 1
net.aq <- nnet(formula.1,data = data1,maxit=300,size=1,linout=T)
summary(net.aq)
yhat.aq4 = predict(net.aq, newdata = data2)
ylhat41=y2-yhat.aq4
#step 2
net.d <- nnet(formula.2,data = data1,maxit=300,size=1,linout=T)
summary(net.d)
yhat.d4=predict(net.d,newdata=data2)
vhat41=d2-yhat.d4

#step 3
net.aq <- nnet(formula.3,data = data2,maxit=300,size=1,linout=T)
summary(net.aq)
yhat.aq4 = predict(net.aq, newdata = data1)
ylhat42=y1-yhat.aq4
#step 4
net.d <- nnet(formula.4,data = data2,maxit=300,size=1,linout=T)
summary(net.d)
yhat.d4=predict(net.d,newdata=data1)
vhat42=d1-yhat.d4


#step5: reg ylhat vhat
lm.fit1=lm(ylhat41~vhat41)
summary(lm.fit1)

lm.fit2=lm(ylhat42~vhat42)
summary(lm.fit2)

# DML2: combine and reg
dim1=length(ylhat41)
dim2=length(ylhat42)
dim=dim1+dim2

dim3=dim1+1
yhat=rep(NA,dim)
yhat[1:dim1]=ylhat41
yhat[dim3:dim]=ylhat42

vhat=rep(NA,dim)
vhat[1:dim1]=vhat41
vhat[dim3:dim]=vhat42

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
cat("Double machine learning 1 (nuetral networks , 2-folds):","\n")
cat("Estimate, s.e., t-statistic, p.value, 95%lower, 95%upper","\n")
print(cbind(theta=be,se.robust=se,t.value=t,pvalue=round(2*(1-pnorm(abs(be/se))),5),
            be-1.96*se,be+1.96*se),digits=4)

cat("t-statistic critial values: 90%=1.65, 95%=1.96, 99%=2.58","\n")
cat("-----------------------------------------------------------","\n")




cat("-----------------------------------------------------------","\n")
cat("Double machine learning 2 (nuetral networks, 2-folds):","\n")
cat("Estimate, s.e., t-statistic, p.value, 95%lower, 95%upper","\n")
print(cbind(theta=beAll,se.robust=seAll,t.value=tAll,pvalue=round(2*(1-pnorm(abs(beAll/seAll))),5),
            beAll-1.96*seAll,beAll+1.96*seAll),digits=4)

cat("t-statistic critial values: 90%=1.65, 95%=1.96, 99%=2.58","\n")
cat("-----------------------------------------------------------","\n")
}

return(list(theta1=be,se.robust1=se, theta2=beAll,se.robust2=seAll))
}



