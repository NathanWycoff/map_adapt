#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  hcr_competitors.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 02.08.2024

# Structured selection with hierarchical models a la Roth and Fischer
import pickle
import numpy as np
import scipy
from matplotlib.gridspec import GridSpec
import pandas as pd
from time import time
import statsmodels.api as sm
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from xgboost import XGBRegressor
from tensorflow_probability.substrates import jax as tfp
import jax.numpy as jnp
from sklearn.neighbors import NearestNeighbors

print(sys.argv)

big_boi = True #Use quadratic model? 
synthetic = False
eu_only = True

exec(open('python/hcr_lib.py').read())
exec(open('python/hcr_settings.py').read())
exec(open('python/glmnet_wrapper.py').read())

# seed = int(sys.argv[1])
seed = 0

np.random.seed(seed+1)
X_train, y_train, X_test, y_test, xcols, re_names, av_names_big = get_data(expansion, synthetic, eu_only, prop_train = prop_train, norm = False)

### XGB
## create model instance
#bst = XGBRegressor(n_estimators=1000, max_depth=5, learning_rate=0.1, objective='count:poisson')
## fit model
#bst.fit(X_train, y_train)
## make predictions
#preds_xgb = bst.predict(X_test)
#nll_xgb = -jnp.sum(tfp.distributions.Poisson(rate=preds_xgb).log_prob(y_test))

## GLMnet
_, preds_glmnet = glmnet_fit(X_train, y_train, X_test, lik = 'poisson')
nll_glmnet = -np.nansum(tfp.distributions.Poisson(rate=preds_glmnet).log_prob(y_test))

## Statsmodels Negative Binomial

## Write output
#resdf = pd.DataFrame([[nll_xgb, nll_glmnet],[seed,seed]]).T
resdf = pd.DataFrame({'nll' : [nll_xgb, nll_glmnet], 'seed' : [seed,seed]})
resdf.index = ['XGB','glmnet']

#simdir = 'sim_out/hcr_eu/' if eu_only else 'sim_out/hcr_global'
outdir = 'sim_out/'+simout_dir

resdf.to_csv(outdir+'comp_'+str(seed)+'.csv')
