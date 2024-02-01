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
import statsmodels.api as sm

import rpy2
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
import rpy2.robjects as robjects

from rpy2.robjects import numpy2ri
from rpy2.robjects import default_converter

test_funcs = {}
r = robjects.r

r['source']('python/R_MLGL_fun.R')
#r['source']('python/R_exclusive_lasso.R')

exec(open('python/sim_lib.py').read())
#exec(open('python/hier_lib.py').read())

def mlgl_fit_pred(X, y, XX, sigma_err, Pu, P, group = 'none', logistic = False):

    if group=='yes':
        groups_R = groups
        var_R = np.arange(P)
    elif group=='hier':
        _, ngroups, P, v1, v2 = hier2nd_sparsity(Pu,1)
        Pi = int(scipy.special.binom(Pu,2))
        var_R = np.repeat(np.nan, 5*ngroups)
        for g in range(ngroups):
            var_R[5*g+0] = v1[g]
            var_R[5*g+1] = v2[g]
            var_R[5*g+2] = Pu+g
            var_R[5*g+3] = Pu+Pi+v1[g]
            var_R[5*g+4] = Pu+Pi+v2[g]
        groups_R = np.repeat(np.arange(ngroups), 5)
    elif group=='none':
        var_R = np.arange(P)
        groups_R = np.arange(P)
    else:
        raise Exception

    var_R += 1
    groups_R += 1

    X_arr = robjects.FloatVector(X.T.flatten())
    X_R = robjects.r['matrix'](X_arr, nrow = X.shape[0])

    y_arr = robjects.FloatVector(y)
    y_R = robjects.r['matrix'](y_arr, ncol = 1)

    var_RR = robjects.IntVector(var_R)
    groups_RR = robjects.IntVector(groups_R)

    mlgl_betas = r['mlgl_fitpred'](X_R, y_R, np.square(sigma_err), var_RR, groups_RR, logistic=logistic)
    mlgl_betahat = mlgl_betas[1:]
    mlgl_beta0 = mlgl_betas[0]

    preds = XX @ mlgl_betahat + mlgl_beta0 
    if logistic:
        preds = preds > 0

    return mlgl_betahat, preds

""" def ex_fit_pred(X, y, XX, sigma_err, groups):
    groups_R = groups+1

    X_arr = robjects.FloatVector(X.T.flatten())
    X_R = robjects.r['matrix'](X_arr, nrow = X.shape[0])

    y_arr = robjects.FloatVector(y)
    y_R = robjects.r['matrix'](y_arr, ncol = 1)

    groups_RR = robjects.IntVector(groups_R)

    mlgl_betas = r['ex_fitpred'](X_R, y_R, np.square(sigma_err), groups_RR)
    mlgl_betahat = mlgl_betas[1:]
    mlgl_beta0 = mlgl_betas[0]

    preds = XX @ mlgl_betahat + mlgl_beta0 

    return mlgl_betahat, preds
 """
