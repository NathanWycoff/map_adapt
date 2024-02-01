#!/usr/bin/Rscript
#  R_MLGL_fun.R Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.19.2023

library(MLGL)

## Fit a group lasso to the data
mlgl_fitpred <- function(X, y, sigma2, var, group, logistic = FALSE) {
    #print(length(var))
    #print(length(group))
    if (logistic) {y <- 2*y-1}
    # Least square loss
    loss <- ifelse(logistic, 'logit','ls')
    res <- overlapgglasso(X, y, var, group, loss=loss)

    fit <- predict(res, X)

    if (logistic) {
        twice_nll <- -2*apply(plogis(fit), 2, function(p) sum(dbinom((1+y)/2,1,0.05+0.9*p,log=TRUE)))
    } else {
        sse <- apply(fit, 2, function(yp) sum((y-yp)^2))
        twice_nll <- 1/(sigma2) * sse
    }
    bic <- twice_nll + log(nrow(X))*res$nVar
    minind <- which.min(bic)

    #preds <- predict(res, XX)[,minind]
    smol_exp_beta <- res$beta[[minind]]
    exp_inds <- as.numeric(substr(names(smol_exp_beta), 2, 100))
    big_exp_beta <- rep(0,length(var))
    big_exp_beta[exp_inds] <- smol_exp_beta 

    beta0 <- res$b0[minind]

    beta_est <- rep(0,ncol(X))
    for (i in 1:nrow(X)) {
        inds <- var==i
        bind <- big_exp_beta[inds]
        bnz <- sum(bind!=0)
        #if (bnz > 0) beta_est[i] <- sum(big_exp_beta[inds]) / bnz
        if (bnz > 0) beta_est[i] <- sum(big_exp_beta[inds]) 
    }

    return(c(beta0, beta_est))
}

#sigma2 <- 1
#set.seed(42)
#N <- 1000
#X <- simuBlockGaussian(N, 12, 5, 0.7)
#XX <- simuBlockGaussian(50, 12, 5, 0.7)
#y <- X[, c(2, 7, 12)] %*% c(2, 2, -2) + rnorm(50, 0, 0.5)
#y <- as.numeric(y > 0)
#var <- c(1:60, 1:8, 7:15)
#group <- c(rep(1:12, each = 5), rep(13, 8), rep(14, 9))
#
#mlgl_fitpred(X, y, sigma2, var, group)
#
