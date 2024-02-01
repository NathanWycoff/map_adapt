#!/usr/bin/Rscript
#  R_MLGL_fun.R Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.19.2023

library(MLGL)

## Fit a group lasso to the data
#mlgl_fitpred <- function(X, y, var, group, logistic = FALSE) {
mlgl_fitpred <- function(X, y, var, group, logistic = FALSE) {
    if (logistic) {y <- 2*y-1}
    # Least square loss
    loss <- ifelse(logistic, 'logit','ls')

    res <- overlapgglasso(X, y, var, group, loss=loss)

    fit <- t(predict(res, X))
    nl <- nrow(fit)

    #preds <- predict(res, XX)[,minind]
    smol_exp_beta <- res$beta
    big_beta <- matrix(NA, nrow = nl, ncol = length(var))
    for (i in 1:nl) {
        exp_inds <- as.numeric(substr(names(smol_exp_beta[[i]]), 2, 100))
        big_exp_beta <- rep(0,length(var))
        big_exp_beta[exp_inds] <- smol_exp_beta[[i]] 
        big_beta[i,] <- big_exp_beta
    }

    beta0 <- res$b0

    #beta_est <- rep(0,ncol(X))
    beta_est <- matrix(0, nrow = nl, ncol = ncol(X))
    for (j in 1:nl) {
        for (i in 1:nrow(X)) {
            inds <- var==i
            bind <- big_beta[j,][inds]
            bnz <- sum(bind!=0)
            #if (bnz > 0) beta_est[i] <- sum(big_exp_beta[inds]) / bnz
            if (bnz > 0) beta_est[j,i] <- sum(big_beta[j,][inds]) 
        }
    }

    return(cbind(beta0, beta_est))
}


