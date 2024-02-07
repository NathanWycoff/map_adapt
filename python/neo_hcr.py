#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  unpop_fit.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.16.2023

# TODO: why now reduction achieves?

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
#seed = int(sys.argv[1])
for i in range(100):
    print("Manual settings!")
seed = 5

# seed = int(time()*100) % 10000
# seed = 1900
# np.random.seed(seed)
np.random.seed(123)
# lik = 'poisson'
lik = 'zinb'
# lik = 'nb'
# lik = 'normal'
use_tr = False
#use_tr = True
use_nest = False
big_boi = True #Use quadratic model? 

sgd = True
synthetic = True
synthetic_interact = True
momentum = False
#momentum = True
#LOG_PROX = False
LOG_PROX = True
if LOG_PROX:
    GLOB_prox = 'log'
else:
    GLOB_prox = 'std'

random_effects = True
#eu_only = False
eu_only = True

exec(open('python/jax_nsa.py').read())
exec(open('python/jax_hier_lib.py').read())
exec(open('python/sim_lib.py').read())
exec(open('python/hcr_settings.py').read())
exec(open('python/glmnet_wrapper.py').read())
key = jax.random.PRNGKey(seed)

df = pd.read_csv('./data/hcr_impu1.csv').iloc[:, 1:]

## Keep only european countries 
europe_iso = [
'ALB','AND','AUT','BLR','BEL','BIH','BGR','HRV','CYP','CZE','DNK','EST','FIN','FRA','DEU','GRC','HUN','ISL','IRL','ITA','KOS','LVA','LIE','LTU','LUX','MKD','MLT','MDA','MCO','MNE','NLD','NOR','POL','PRT','ROU','RUS','SMR','SRB','SVK','SVN','ESP','SWE','CHE','TUR','UKR','GBR','VAT'
]
#europe_iso = europe_iso[:(len(europe_iso)//5)]
#europe_iso = europe_iso[:(len(europe_iso)//2)]

df['dist'] = np.log(df['dist'])
eu2eu = np.logical_and(df.iso_d.isin(europe_iso), df.iso_o.isin(europe_iso))
if eu_only:
    df = df.loc[eu2eu,:]

realx = 'dist'
if synthetic:
    if synthetic_interact:
        lmu = df['pop_o']/np.max(df['pop_o']) + df['dist'] / np.max(df['dist']) - df['pop_o']/np.max(df['pop_o'])*df['dist'] / np.max(df['dist'])
        lmu *= 10
        if not big_boi:
            print("Warning: using interaction data in main effects model.")
    else:
        lmu = df['pop_o']/np.max(df['pop_o']) + df['dist'] / np.max(df['dist'])
        lmu *= 5
    y_dist = tfpd.Poisson(log_rate=lmu)
    y = y_dist.sample(seed=key)
    print(np.max(y))
else:
    y = np.array(df['newarrival'])
    if lik == 'normal':
        y = np.log(y+1)

marg_vars = [x for x in df.columns[3:] if (
    x[-2:] in ['_o', '_d'] and x not in ['Country_o', 'Country_d'])]
dy_vars = list(df.columns[-8:])
xcols = np.array(marg_vars+dy_vars)
Xd = df[xcols]

X = np.array(Xd)
Xi = X.copy()
X = (X - np.mean(X, axis=0)[np.newaxis, :]) / np.std(X, axis=0)[np.newaxis, :]

n_train = int(np.ceil(prop_train * X.shape[0]))

if random_effects:
    B = df.loc[:, ['iso_o', 'iso_d', 'year']]
    B['year'] = B['year'].astype(str)
    # Bd = pd.get_dummies(B, drop_first=True)
    Bd = pd.get_dummies(B, drop_first=False)
    X = np.concatenate([X, Bd], axis=1)
    # dont_pen = np.arange(Xi.shape[1], X.shape[1])
    # if lik == 'zinb':
    #    dont_pen = np.concatenate([dont_pen, dont_pen+X.shape[1]])
dont_pen = np.array([]).astype(int)

re_names = [x+'_o' for x in list(set(B['iso_o']))] + \
    [x+'_d' for x in list(set(B['iso_d']))]+list(set(B['year']))
av_names = np.concatenate([xcols, re_names])

# subsize = 200
# subsize = np.inf

#l2_coef = 1e-2  # TODO: Right value?
l2_coef = 0.  # TODO: Right value?
# l2_coef = 0.  # TODO: Right value?
# l2_coef=1e-1# TODO: Right value?
verbose = False
use_hier = True

# tau0 = 1e6  # 440 nz
# tau0 = 1e7
# tau0 = np.power(seed,10)

#tau0 = np.logspace(5, 6, reps)[seed]
tau0 = 1e4
#tau0 = 1e5

# if random_effects:
#    groups = np.concatenate(
#        [np.arange(Xi.shape[1]), np.repeat(Xi.shape[1], X.shape[1] - Xi.shape[1])])
#    groups = np.concatenate([groups, groups])
# else:
#    gg = np.arange(Xi.shape[1])
#    groups = np.concatenate([gg, gg])

Xempty = X[:0, :]
Xempty_big = add_int_quad(Xempty, var_names=list(av_names))
n_re = len(set(df['iso_d'])) + len(set(df['iso_d'])) + len(set(df['year']))
#Xempty_big = Xempty_big.iloc[:, :-n_re]
av_names_big = Xempty_big.columns
# Xd
yempty = np.array([])

me_names = av_names_big[:X.shape[1]]
int_names = av_names_big[X.shape[1]:-X.shape[1]]
qu_names = av_names_big[-Xd.shape[1]:]

if use_hier:
    Pu = Xempty.shape[1]
    Pnz = 1
    _, ngroups, P, v1, v2 = hier2nd_sparsity(Pu, Pnz)

    Pi = int(scipy.special.binom(Pu, 2))
    Pq = Pu
    # P = Pu + Pi + Pq

# def adaptive_prior(x, pv, mod):
#    # lam_dist = tfpd.Cauchy(loc=1., scale=mod.P)
#    #lam_dist = tfpd.Cauchy(loc=1., scale=1)
#    lam_dist = tfpd.Cauchy(loc=X.shape[0], scale=1)
#    lam_dens = -jnp.sum(lam_dist.log_prob(x)-jnp.log((1-lam_dist.cdf(0))))
#    # lam_dens = -jnp.sum(lam_dist.log_prob(x))
#    return lam_dens

Pme = len(xcols)
Pre = len(re_names)
Pme_me = int(scipy.special.binom(Pme, 2))
Pre_re = int(scipy.special.binom(Pre, 2))
Pii = int(scipy.special.binom(Pre+Pme, 2))

av = Xempty_big.columns
x_me = av[:Pme] #X
r_me = av[Pme:(Pme+Pre)] #r
all_i = av[(Pme+Pre):(Pme+Pre+Pii)]

#re_targets = dict([(re,[]) for re in re_names])
re_invdict = {}
re_invmap1 = np.zeros(len(av_names_big)).astype(int)-1
re_invmap2 = np.zeros(len(av_names_big)).astype(int)-1
for ii,i in enumerate(av_names_big):
    res_in = []
    for ri,re in enumerate(re_names):
        if re in i:
            res_in.append(ri)
        if len(res_in)==0:
            pass
        elif len(res_in)==1:
            re_invmap1[ii] = re_invmap2[ii] = res_in[0]
        elif len(res_in) == 2:
            re_invmap1[ii] = res_in[0]
            re_invmap2[ii] = res_in[1]
        else:
            raise Exception()

re_invmap1 += 1
re_invmap2 += 1
#np.sum(re_invmap1==0)
#np.sum(re_invmap2==0)

# Standard hurdle prior
def hier_prior_hurdle(x, pv, mod):
    gamma = jnp.exp(pv['log_gamma'])
    gamma_dist = tfpd.Cauchy(loc=mod.N, scale=1.)
    gamma_dens = -jnp.sum(gamma_dist.log_prob(gamma) -
                          jnp.log((1-gamma_dist.cdf(0))))

    lam_sd = 1.

    GAMMAt = make_gamma_mat(Pu).astype(int)
    gam_meq = jnp.min(gamma[GAMMAt], axis=0)
    # Hurdle model shares
    gam_meq = jnp.repeat(gam_meq, 2)
    meq_dist = tfpd.Normal(loc=gam_meq, scale=lam_sd)
    hurd_gamma = jnp.repeat(gamma, 2)
    i_dist = tfpd.Normal(loc=hurd_gamma, scale=lam_sd)

    me_vanil = np.arange(Pu)
    me_inds = np.concatenate([me_vanil, P+me_vanil])
    q_vanil = np.arange(Pu+Pi, Pu+Pi+Pq)
    q_inds = np.concatenate([q_vanil, P+q_vanil])
    i_vanil = np.arange(Pu, Pu+Pi)
    i_inds = np.concatenate([i_vanil, P+i_vanil])

    lam_me = x[me_inds]
    lam_q = x[q_inds]
    me_dens = -jnp.sum(meq_dist.log_prob(lam_me)-jnp.log(1-meq_dist.cdf(0)))
    q_dens = -jnp.sum(meq_dist.log_prob(lam_q)-jnp.log(1-meq_dist.cdf(0)))

    lam_i = x[i_inds]
    i_dens = -jnp.sum(i_dist.log_prob(lam_i)-jnp.log(1-i_dist.cdf(0)))

    return gamma_dens + me_dens + q_dens + i_dens

# x = mod_big.vv['lam']
# pv = mod_big.vv
# mod = mod_big

mnll_smol = np.zeros(reps)
mnll_big = np.zeros(reps)
# for rep in range(reps):
inds_train = np.random.choice(X.shape[0], n_train)
inds_test = np.delete(np.arange(X.shape[0]), inds_train)
X_train = X[inds_train, :]
if not sgd:
    X_big_train = np.array(X_big)[inds_train,:]
y_train = y[inds_train]
X_test = X[inds_test, :]
y_test = y[inds_test]

log_gamma = jnp.zeros(Pi)
lam_prior_vars = {'log_gamma': log_gamma}
#mod_big = jax_vlMAP(X_train, y_train, hier_prior_hurdle, lam_prior_vars, lik=lik, tau0=tau0, track=False, l2_coef=l2_coef, logprox=LOG_PROX, quad = True)
mod_big = jax_vlMAP(X_train, y_train, adaptive_prior, lam_prior_vars, lik=lik, tau0=tau0, track=False, l2_coef=l2_coef, logprox=LOG_PROX, quad = True)
print("Marginal Prior!")
mod_big.fit(max_iters=1000, conv_thresh = -1)

mod_big.plot()

np.sum(mod_big.vv['beta']!=0)

mean_func = np.where(mod_big.vv['beta'][:Xempty_big.shape[1]]!=0)[0]
print(av_names_big[mean_func])
zero_func = np.where(mod_big.vv['beta'][Xempty_big.shape[1]:]!=0)[0]
print(av_names_big[zero_func])

df = pd.DataFrame([mod_big.vv['beta'][mean_func], av_names_big[mean_func]]).T
df.sort_values(0)
