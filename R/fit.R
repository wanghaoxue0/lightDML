# fit function for lightDML
# Haoxue Wang 6 Feb. 2023
# hw613@cam.ac.uk

fit <- function(data, y, x, d, z=NA, ml=c("bagging","boosting","random forest", "neural network"), score=c("ATE, LATE"), random_state=2023, verbose=FALSE){
  if (score=="ATE"){
    if (ml=="bagging"){
      model = bagging(data, y, x, d, z, score=score, random_state=random_state, verbose=verbose)
    } else if(ml=="boosting"){
      model = boosting(data, y, x, d, z, score=score, random_state=random_state, verbose=verbose)
    } else if(ml=="random forest"){
      model = rf(data, y, x, d, z, score=score, random_state=random_state, verbose=verbose)
    } else if(ml=="neural network"){
      model = nn(data, y, x, d, z, score=score, random_state=random_state, verbose=verbose)
    } else{
      cat("please specify the right machine learning algorithm")
    }
  } else if(score=="LATE"){
    if (ml=="bagging"){
      ml_g = lrn("regr.ranger", mtry = ncol(data)-3, min.node.size = 2, max.depth = 5)
      ml_m = lrn("classif.ranger", mtry = ncol(data)-3, min.node.size = 2, max.depth = 5)
      ml_r = ml_m$clone()
    } else if(ml=="boosting"){
      ml_g = lrn("regr.xgboost")
      ml_m = lrn("classif.xgboost")
      ml_r = ml_m$clone()
    } else if(ml=="random forest"){
      ml_g = lrn("regr.ranger", num.trees = 25, min.node.size = 2, max.depth = 5)
      ml_m = lrn("classif.ranger", num.trees = 25, min.node.size = 2, max.depth = 5)
      ml_r = ml_m$clone()
    } else if(ml=="neural network"){
      ml_g = lrn("regr.nnet")
      ml_m = lrn("classif.nnet")
      ml_r = ml_m$clone()
    } else{
      cat("please specify the right machine learning algorithm")
    }
    obj_dml_data = DoubleMLData$new(data, y_col="y", d_cols="d", z_cols="z")
    dml_iivm_obj = DoubleMLIIVM$new(obj_dml_data, ml_g, ml_m, ml_r)
    dml_iivm_obj$fit()
    list(dml_iivm_obj$all_coef,dml_iivm_obj$all_se)
  }
}
