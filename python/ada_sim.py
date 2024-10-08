#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  ada_sim.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.18.2023

## Check the performance of our adaptive selection across:
# TODO
# 1) ida_net cross validation
# 2) floor generated dense beta may not be what we want for group models.

## Structured selection with hierarchical models a la Roth and Fischer
import pickle
import numpy as np
import scipy
from matplotlib.gridspec import GridSpec
import pandas as pd
from time import time
import statsmodels.api as sm
import sys

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

#exec(open('python/nonsmooth_approx_lib.py').read())
#exec(open('python/nsa_foyer.py').read())
exec(open('python/jax_nsa.py').read())
exec(open('python/jax_hier_lib.py').read())
exec(open('python/sim_lib.py').read())
exec(open('python/sim_settings.py').read())
exec(open('python/MLGL_wrapper.py').read())
exec(open('python/glmnet_wrapper.py').read())
#exec(open('python/ida_load.py').read())
#exec(open('python/jags_horseshoe.py').read())

def null_pred():
    beta_hat = np.repeat(np.nan, P)
    preds = np.repeat(np.nan, NN)
    return beta_hat, preds

#manual = True
manual = False

verbose = True
LOG_PROX = True

l2_coef = 0.

nrec = 3

nqoi = len(qoi)
ncomps = len(models2try)
#res = pd.DataFrame(np.zeros([ncomps*iters*nset,nqoi+nrec+1+1+1])) # The 1 is the competitor and 1 is the setting, and 1 is the execution time.
res = pd.DataFrame(np.zeros([ncomps,nqoi+nrec+1+1+1+1])) # The 1 is the competitor and 1 is the setting, and 1 is the seed, and 1 is the execution time.
# TODO: Add iters, execution time.
res.columns = ['Method','Setting','Seed','Time']+['N','P','rho_P']+qoi

def eval_mod(betahat, preds): #TODO: libsvm
    if lik in ['normal','poisson','nb']:
        yymse = np.mean(np.square(yy-preds))
    elif lik == 'bernoulli':
        if np.any(np.isnan(preds)):
            yymse = np.nan
        else:
            yymse = np.mean(yy!=preds) # TODO: Predictive log loglikelihood?
    elif lik == 'cauchy':
        yymse = np.median(np.abs(yy-preds))
    
    if sim=='synthetic':
        mseb = np.mean(np.square(beta-betahat))
        fpr = np.mean(betahat[~nonzero.astype(bool)]!=0)
        fnr = np.mean(betahat[nonzero.astype(bool)]==0)
        return mseb, yymse, fpr, fnr
    else:
        nz = np.sum(betahat!=0)
        return yymse, nz

if manual:
    if len(sys.argv)>1:
        for i in range(10):
            print("Passed arguments but manual mode is on!")
        quit()
    for i in range(10):
        print("Manual")
    #s_i = '0'
    #s_i = 'bike'
    #s_i = 'obesity'
    s_i = 'seoul'
    #s_i = 'abalone'
    #s_i = 'kin40k'
    #s_i = 'keggu'
    #lr = 5e-6
    #lr = 1e-5
    #lr = 1e-5
    #lr = 1e-5
    lr = 1e-3
    ada = True
    #max_iters = 1000
    models2try = ['sbl_ada','OLS','glmnet']
    #s_i = 'year'
    #s_i = 'heart'
    #s_i = 'diabetes'
    #s_i = '3'
    seed = 0
    #models2try = ['sbl_hier','glmnet']
else:
    print(sys.argv)
    s_i = sys.argv[1]
    seed = int(sys.argv[2])

s_i_seed = sum([ord(a) for a in s_i])

key = jax.random.PRNGKey(seed+s_i_seed)
np.random.seed(seed+s_i_seed)

if sim == 'synthetic':
    s_i = int(s_i)
    N, Pu, Pnz, lik, sigma_err = settings[s_i]

    if sparsity_type == 'random':
        P = Pu
        nonzero = random_sparsity(P,Pnz)
        X_all = np.random.normal(size=[N+NN,P])
    elif sparsity_type=='group':
        Ppg = 5
        #G = np.random.poisson(Ppg-2,size=Pu)+2
        G = np.repeat(Ppg, Pu)
        P = int(np.sum(G))
        X_all = np.random.normal(size=[N+NN,P])
        groups = np.repeat(np.arange(Pu), G)
        members = [np.where(groups==i)[0] for i in range(Pu)]
        gsize = np.repeat(G, G)

        nz_groups = np.random.choice(Pu, Pnz, replace = True)
        nonzero = np.concatenate([(m in nz_groups) * np.ones(G[m]) for m in range(Pu)])

    elif sparsity_type=='hier2nd':
        Xu_all = np.random.normal(size=[N+NN,Pu])
        Xdf_all = add_int_quad(Xu_all)
        X_all = np.array(Xdf_all)
        nonzero, ngroups, P, v1, v2 = hier2nd_sparsity(Pu,Pnz)

        Pi = int(scipy.special.binom(Pu,2))
        Pq = Pu

    y_all = np.zeros(shape=N)
    while np.sum(y_all != 0) < len(y_all)//10:
        if beta_style=='random':
            beta_dense = np.random.normal(size=P)
        elif beta_style=='floor':
            beta_pre = np.random.normal(size=P)
            beta_dense = np.sign(beta_pre)*(0.5 + np.abs(beta_pre))
        else:
            beta_dense = np.concatenate([2*np.ones(Pu), np.ones(Pi), np.ones(Pq)]) * np.random.choice([-1,1],P)
    
        beta = beta_dense * nonzero
        #if lik in ['nb','poisson','bernoulli']:
        #    beta /= np.max(np.abs(beta))
        mu_y = X_all @ beta
        if lik in ['nb','poisson','bernoulli']:
            #beta = 10 * beta / np.max(mu_y)
            beta = 5 * beta / np.max(mu_y)
            mu_y = X_all @ beta
        if lik == 'normal':
            dist_y = tfpd.Normal(loc=mu_y, scale = sigma_err)
        elif lik == 'cauchy':
            dist_y = tfpd.Cauchy(loc=mu_y, scale = sigma_err)
            #tau_range = [5e1, 1e3]
        elif lik == 'poisson':
            dist_y = tfpd.Poisson(log_rate=mu_y)
        elif lik == 'nb':
            a = 1/np.square(sigma_err)
            dist_y = tfpd.NegativeBinomial(total_count=a, logits=mu_y)
        elif lik == 'bernoulli':
            dist_y = tfpd.Bernoulli(logits=mu_y)
    
        y_all = np.array(dist_y.sample(seed=key))
    
    X = X_all[:N,:]
    if sparsity_type=='hier2nd':
        Xu = Xu_all[:N,:]
    XX = X_all[N:(N+NN),:]
    y = y_all[:N]
    yy = y_all[N:(N+NN)]
elif sim == 'libsvm':
    #with open("pickles/"+s_i+".pkl",'rb') as f:
    #    X_all, y_all = pickle.load(f)
    df = pd.read_csv(data_dir+s_i+'.csv')
    X_all = np.array(df.iloc[:,:-1])
    y_all = np.array(df['y'])

    N,Pu = X_all.shape

    if sparsity_type=='hier2nd':
        Xu_all = np.copy(X_all)
        Xdf = add_int_quad(X_all)
        X_all = np.array(Xdf)
        #TODO: Duplicated code.
        nonzero, ngroups, P, v1, v2 = hier2nd_sparsity(Pu,Pnz)
        Pi = int(scipy.special.binom(Pu,2))
        Pq = Pu
    elif sparsity_type=='random':
        X_all = np.array(X_all)
        P = Pu
    else:
        raise Exception("hier2 and random only allowed for libsvm.")

    lik = liks[s_i]

    sigma_err = np.var(y_all)/10

    test_set = np.sort(np.random.choice(N, N//2, replace = False))
    train_set = np.array(list(set(np.arange(N)).difference(test_set)))
    if sparsity_type=='hier2nd':
        XX = X_all[test_set,:]
        XXu = Xu_all[test_set,:]
    yy = y_all[test_set]
    X = X_all[train_set,:]
    Xu = Xu_all[train_set,:]
    y = y_all[train_set]
    NN = len(test_set)
    N = len(train_set)
else:
    raise Exception("Dataset not recognized.")

if sim=='synthetic':
    if sparsity_type=='random':
        #TAU0 = 0.1 * N
        #TAU0 = 0.015*N # Good for s_i=0,2
        TAU0 = 0.025*N # Good for s_i=0,2
    elif sparsity_type=='group':
        #TAU0 = 0.020*N # Good for s_i=0,2
        print("new tau:")
        TAU0 = 0.015*N # Good for s_i=0,2
    elif sparsity_type=='hier2nd':
        #TAU0 = 0.020*N # Good for s_i=0,2
        print("new tau:")
        TAU0 = 0.015*N # Good for s_i=0,2
    else:
        raise NotImplementedError
elif sim=='libsvm':
    assert sparsity_type=='hier2nd'
    #TAU0 = 1*N
    TAU0 = 0.015*N
    #TAU0 = 0.15*N # Abalone and bike liked this one.
    #TAU0 = 1.5*N
    #TAU0 = 0.5*N
    print("Changed status")
else:
    raise Exception()

####
####
#ret = np.linalg.svd(X, full_matrices = False)
#ret[1]

###############
###############

for ind, modname in enumerate(models2try):
    tt = time()
    if verbose:
        print(modname)

    # Our models
    if modname[:4]=='sbl_':
        if modname[4:] == 'ada':
            lam_prior_vars = {}
            prior = adaptive_prior
        elif modname[4:]=='group':
            lam_prior_vars = {'log_gamma' : jnp.zeros(Pu)}
            prior = group_prior
        elif modname[4:]=='hier':
            lam_prior_vars = {'log_gamma' : jnp.zeros(Pi)}
            prior = hier_prior
        else:
            raise Exception("Modname not found.")
        nzparams = 0
        nz_counter = 0
        while nzparams == 0:
            mod = jax_vlMAP(X, y, prior, lam_prior_vars, lik = lik, tau0 = TAU0*np.power(0.5,nz_counter), track = manual, mb_size = mb_size, logprox=LOG_PROX, es_patience = es_patience, l2_coef = l2_coef)
            mod.fit(max_iters=max_iters, verbose=True, lr_pre = lr, ada = ada)
            nzparams = np.sum(mod.beta!=0)
            nz_counter += 1
            if nzparams==0:
                print("No nonzeros; restarting.")
        #mod.fit(c_relax = 0.5, max_iters = max_iters, pc = 'identity')

        if manual:
            mod.plot()
        else:
            mod.plot('./debug_out/'+modname+'_'+lik+'_'+str(seed)+'.png')

        #with open('yeboi.pkl', 'wb') as f:
        #    pickle.dump([X,y], f)

        beta_hat = np.array(mod.beta)
        if lik in ['poisson','bernoulli','nb']:
            preds = np.array(mod.predictive(XX).mean())
        else:
            preds = np.array(mod.predictive(XX).loc)
        if lik == 'bernoulli':
            preds = preds > 0.5

    elif modname == 'glmnet':
        if lik in ['normal','bernoulli','poisson','nb']:
            likglm = 'poisson' if lik=='nb' else lik
            beta_hat, preds = glmnet_fit(X, y, XX, lik = likglm)
            if lik=='bernoulli':
                preds = preds > 0
        else:
            beta_hat, preds = null_pred()
        
    elif modname == 'jags':
        if N <= JAGS_MAX_N or seed < JAGS_N_REPS:
            #for i in range(10):
            #    print("Waring: smol JAGS!")
            beta_hat, preds =hs_est(X, y, XX, lik, samples = 1000)
            if lik == 'bernoulli':
                preds = preds > 0.5
        else:
            beta_hat, preds = null_pred()

    elif modname in ['lasso','MLGL']: # Lassos
        if not (modname=='MLGL' and s_i=='spam' and seed > 4):
            if modname == 'lasso':
                grouped = False
            elif modname == 'MLGL':
                grouped = True
            else:
                raise Exception
            if lik in ['normal','bernoulli']:
                if grouped:
                    if sparsity_type=='group':
                        group='yes'
                    elif sparsity_type=='hier2nd':
                        group='hier'
                    else:
                        raise Exception
                else:
                    group='none'
                beta_hat, preds = mlgl_fit_pred(X, y, XX, sigma_err, Pu, P, group = group, logistic = lik=='bernoulli')
            else:
                beta_hat, preds = null_pred()
        else:
            beta_hat, preds = null_pred()
    elif modname == 'ida_net':
        if lik == 'normal':
            #beta_hat, preds = ida_net(X, y, XX, groups, alpha = 0.0)
            beta_hat, preds = ida_net_sc(X, y, XX, groups, alpha = 0.5, Q = 10)
        else:
            beta_hat, preds = null_pred()

    elif modname=='OLS':
        if P<N:
            try:
                if lik == 'normal':
                    fit = sm.OLS(y, sm.add_constant(X)).fit()
                elif lik == 'bernoulli':
                    fit = sm.GLM(y, sm.add_constant(X), family = sm.families.Binomial()).fit()
                elif lik == 'poisson':
                    fit = sm.GLM(y, sm.add_constant(X), family = sm.families.Poisson()).fit()
                elif lik == 'nb':
                    fit = sm.GLM(y, sm.add_constant(X), family = sm.families.NegativeBinomial()).fit()
                elif lik == 'cauchy':
                    fit = sm.RLM(y, sm.add_constant(X), M=sm.robust.norms.HuberT()).fit()

                preds = fit.predict(sm.add_constant(XX))
                beta_hat = fit.params[1:]
                if lik=='bernoulli':
                    preds = preds >= 0.5
            except Exception:
                beta_hat, preds = null_pred()
        else:
            beta_hat, preds = null_pred()
    else:
        raise Exception(modname+" not recognized.")

    if verbose:
        print(beta_hat[beta_hat!=0])

    td = time()-tt
    res.iloc[ind,0] = modname
    res.iloc[ind,1] = s_i
    res.iloc[ind,2] = seed
    res.iloc[ind,3] = td
    res.iloc[ind,4] = N
    res.iloc[ind,5] = P
    res.iloc[ind,(4+nrec):(nqoi+nrec+4)] = eval_mod(beta_hat, preds)

if manual:
    for i in range(10):
        print("Manual")
else:
    info_str = 'setting_' + str(s_i) + '_seed_' + str(seed)
    fname = 'sim_out/'+simout_dir+'/'+info_str+'.csv'
    res.to_csv(fname, index = False)

