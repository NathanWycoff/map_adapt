#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  init_debug.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 02.10.2024

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

print(sys.argv)

manual = True
#manual = False

exec(open('python/jax_nsa.py').read())
exec(open('python/jax_hier_lib.py').read())
exec(open('python/hcr_lib.py').read())
exec(open('python/hcr_settings.py').read())
exec(open('python/glmnet_wrapper.py').read())

key = jax.random.PRNGKey(123)

N = 1000
P = 0
X = np.random.normal(size=[N,P])

stddev = jnp.exp(1/2)

#preds_y = 5.
#preds_z = 0.
#
#dist_nz = tfpd.NegativeBinomial(total_count=1/jnp.square(stddev), logits = preds_y)
#dist_z = tfpd.Deterministic(loc=jnp.zeros_like(preds_y))
#pnz = tfpd.Categorical(logits=jnp.stack([preds_z,-preds_z]).T)
##sp = jax.nn.sigmoid(preds_z)
##pnz = tfpd.Categorical(probs=jnp.stack([sp,1.-sp]).T)
#dist_pred = tfpd.Mixture(pnz, components=[dist_z, dist_nz])
#
#y = dist_pred.sample(N, seed = key)

y = np.round(np.abs(np.random.normal(size=N) / np.random.normal(size=N)))
y *= np.random.choice([0,1],N)

verbose = True

mod = jax_vlMAP(X, y, adaptive_prior, {}, lik = 'zinb', tau0 = 1., track = True)

mod.fit(max_iters=2000, prefit = True, verbose=verbose, lr_pre = 0.1, ada = ada, warm_up = True)

mod.plot()


