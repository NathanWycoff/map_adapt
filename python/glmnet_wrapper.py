#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  MLGL_wrapper.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.20.2023

## Structured selection with hierarchical models a la Roth and Fischer
import pickle
import numpy as np
import scipy
from matplotlib.gridspec import GridSpec
import pandas as pd
from time import time

#import os
#os.environ["R_HOME"] = r"/home/nate/R_vanil/R-4.3.1/" # change as needed
import rpy2
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
import rpy2.robjects as robjects

from rpy2.robjects import numpy2ri
from rpy2.robjects import default_converter

from rpy2.robjects.packages import importr
glmnet = importr("glmnet")

test_funcs = {}
r = robjects.r

## POMP 10
#r['source']('python/R_MLGL_fun.R')
#r['source']('python/R_exclusive_lasso.R')

exec(open('python/sim_lib.py').read())
#exec(open('python/hier_lib.py').read()

def glmnet_fit(X, y, XX, lik, taus = None):
    family = None
    if lik == 'normal':
        family = 'gaussian'
    elif lik == 'poisson':
        family = 'poisson'
    elif lik == 'bernoulli':
        family = 'binomial'
    else:
        raise Exception("Unkown family in glmnet_fit.")

    X_arr = robjects.FloatVector(X.T.flatten())
    X_R = robjects.r['matrix'](X_arr, nrow = X.shape[0])

    XX_arr = robjects.FloatVector(XX.T.flatten())
    XX_R = robjects.r['matrix'](XX_arr, nrow = XX.shape[0])

    y_arr = robjects.FloatVector(y)
    y_R = robjects.r['matrix'](y_arr, ncol = 1)

    if taus is None:
        fit = glmnet.cv_glmnet(X_R, y_R, family = family)
        betahat = r['as.matrix'](r['coef'](fit))[1:].flatten()
        preds = r['predict'](fit, XX_R).flatten()
    else:
        fit = glmnet.glmnet(X_R, y_R, family = family, alpha = 1., **{'lambda' : taus})
        betahat = r['as.matrix'](r['coef'](fit))[1:,:].T
        preds = r['predict'](fit, XX_R).T

    #print('a')
    #r['rm']('fit')
    #print('a')
    #r['rm']('X_arr')
    #r['rm']('X_R')
    #print('a')
    #r['rm']('XX_arr')
    #r['rm']('XX_R')
    #print('a')
    #r['rm']('y_arr')
    #r['rm']('y_R')
    #print('a')
    #r['gc']()
    del fit, X_arr, X_R, XX_arr, XX_R, y_arr, y_R
    r['gc']()
    r['gc']()

    #if lik == 'bernoulli':
    #    preds = preds > 0.5
    #elif lik == 'poisson':
    #    preds = np.exp(preds)

    return betahat, preds
