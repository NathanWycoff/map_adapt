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

print(sys.argv)

manual = True
#manual = False

exec(open('python/jax_nsa.py').read())
exec(open('python/jax_hier_lib.py').read())
exec(open('python/hcr_lib.py').read())
exec(open('python/hcr_settings.py').read())
exec(open('python/glmnet_wrapper.py').read())

lik = 'zinb'
#lik = 'normal'

verbose = True
use_hier = big_boi

LOG_PROX = True
if LOG_PROX:
    GLOB_prox = 'log'
else:
    GLOB_prox = 'std'

lr = 1e-3

if manual:
    if len(sys.argv)>1:
        for i in range(10):
            print("Manual settings but arguments provided!")
        quit()

    seed = 0
    #tau_ind = 9
    #tau0 = 1e10
    #tau0 = 1e6
    #max_iters = 100
else:
    tau_ind = int(sys.argv[1])
    seed = int(sys.argv[2])
    #tau0 = np.logspace(0,4,num=n_tau)[tau_ind]
    #tau0 = np.logspace(4,6,num=n_tau)[tau_ind]
    #tau0 = np.logspace(4,np.log10(16681.005372),num=n_tau)[tau_ind]
    #tau0 = np.logspace(2,4,num=n_tau)[tau_ind]
    #tau0 = np.logspace(2,np.log10(774.263683),num=n_tau)[tau_ind]
l2_coef = 0.

#simid = str(tau_ind)+'_'+str(seed)
simid = str(seed)

key = jax.random.PRNGKey(seed)
np.random.seed(seed+1)
X_train, y_train, X_test, y_test, xcols, re_names, av_names_big = get_data(big_boi, synthetic, eu_only)

## Set up hierarchical model.
if use_hier:
    Pu = X_train.shape[1]
    Pnz = 1
    _, ngroups, P, v1, v2 = hier2nd_sparsity(Pu, Pnz)

    Pi = int(scipy.special.binom(Pu, 2))
    Pq = Pu
    # P = Pu + Pi + Pq

Pme = len(xcols)
Pre = len(re_names)
Pme_me = int(scipy.special.binom(Pme, 2))
Pre_re = int(scipy.special.binom(Pre, 2))
Pii = int(scipy.special.binom(Pre+Pme, 2))

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

if use_hier:
    log_gamma = jnp.zeros(Pi)
    lam_prior_vars = {'log_gamma': log_gamma}
    prior = hier_prior_hurdle
else:
    print("Marginal Prior!")
    lam_prior_vars = {}
    prior = adaptive_prior

tau_use = np.flip(tau_range)
#tau_use = tau_range

tau_try = tau_range[-1]
mod = jax_vlMAP(X_train, y_train, prior, lam_prior_vars, lik = lik, tau0 = 1., track = manual, mb_size = mb_size, logprox=LOG_PROX, es_patience = es_patience, quad = big_boi, l2_coef = l2_coef)
nlls = np.zeros(n_tau)

## DRY1
mod.fit(max_iters=max_iters, verbose=False, lr_pre = 1e-1, ada = ada, warm_up = True, prefit = True)
## DRY1
# prefit
mod.plot('prefit.png')

#mod.set_tau0(1e3)
mod.set_tau0(5e2)
## DRY1
mod.fit(max_iters=max_iters, verbose=False, lr_pre = 1e-4, ada = ada, warm_up = True)
## DRY1
mod.plot('fit.png')
print(np.sum(mod.vv['beta']!=0))

df_means = []
df_zeros = []
for ti,tau0 in enumerate(tqdm(tau_use)):
    print(tau0)
    mod.set_tau0(tau_try)
    ## DRY1
    mod.fit(max_iters=max_iters, verbose=False, lr_pre = lr, ada = ada, warm_up = True)
    ## DRY1

    mod.plot('debug_out/'+'hcr_'+str(eu_only)+'_'+str(np.round(tau0))+'.png')

    nlls[ti] = mod.big_nll(X_test, y_test)

    P = len(mod.vv['beta'])
    P2 = P//2
    mean_func = np.where(mod.vv['beta'][:P2]!=0)[0]
    zero_func = np.where(mod.vv['beta'][P2:]!=0)[0]

    df_mean = pd.DataFrame([mod.vv['beta'][mean_func], av_names_big[mean_func]]).T
    df_zero = pd.DataFrame([mod.vv['beta'][P2+zero_func], av_names_big[zero_func]]).T
    df_means.append(df_mean)
    df_zeros.append(df_zero)

    print("NZ:")
    print(df_mean.shape[0] + df_zero.shape[0])

    mod.nll_es

resdf = pd.DataFrame({'nll' : nlls, 'tau' : tau_use})
resdf['nnz'] = [df_means[i].shape[0] + df_zeros[i].shape[0] for i in range(n_tau)]
fname = 'sim_out/'+simout_dir+simid
if not manual:
    #df_mean.to_csv(fname+'_betas_mean.csv')
    #df_zero.to_csv(fname+'_betas_zero.csv')
    resdf.to_csv(fname+'_zinb_nll.csv')
    
    with open("pickles/traj_hcr_"+str(eu_only)+'.pdf', 'wb') as f:
        pickle.dump([df_means, df_zeros, resdf], f)
