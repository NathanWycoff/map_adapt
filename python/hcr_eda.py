#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  unpop_fit.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.16.2023

# List:
# big boi = True (and synth_int=True)
# global
# real data
# zinb

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
import jax

exec(open('python/jax_nsa.py').read())
exec(open('python/jax_hier_lib.py').read())
exec(open('python/hcr_lib.py').read())
exec(open('python/hcr_settings.py').read())
exec(open('python/glmnet_wrapper.py').read())

big_boi = False
synthetic = False
eu_only = True

## Load the data
seed = 123
key = jax.random.PRNGKey(seed)
np.random.seed(seed+1)
X_train, y_train, X_test, y_test, xcols, re_names, av_names_big = get_data(big_boi, synthetic, eu_only, prop_train = 1., norm = False)

df = pd.DataFrame(X_train)
df.columns = list(xcols) + re_names
df.insert(loc=0, column = 'flow', value = y_train)

## Extract marginal vars.
nv = len(xcols)

nr = int(np.ceil(nv/4))

trans = lambda x: np.log10(x+1.)
plt.figure(figsize=[10, 1.5*nr])
cc = []
cc_log = []
for vi, v in enumerate(xcols):
    plt.subplot(nr, 4, vi+1)
    plt.hist(df[v])
    plt.title(v)

    cc.append(np.corrcoef(df[v], trans(df['flow']))[0,1])
    cc_log.append(np.corrcoef(np.log10(df[v]+1), trans(df['flow']))[0,1])
plt.tight_layout()
plt.savefig("marg.png")
plt.close()

cc = pd.Series(cc, index = xcols)
cc_log = pd.Series(cc_log, index = xcols)

dfd = pd.DataFrame({'cc':cc,'cl':cc_log})
dfd['ratio'] = np.square(dfd['cc'] / dfd['cl'])
dfd.sort_values('ratio')

cc_log.sort_values()
cc.sort_values()

letslog = ['area','pop','best_est_o','Nyear_conflict']
