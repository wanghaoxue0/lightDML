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

