#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  jax_nsa.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.31.2023

#exec(open('python/fqs.py').read())

# TODO:
# Proximalize sigma2 log term as well?

# To double check:
#1) that logprox works without pc every iter.
#2) Is it slower? Or only from a movement-break perspective?

import pickle
import numpy as np
import scipy
from matplotlib.gridspec import GridSpec
import pandas as pd
from time import time
import jax
import jax.numpy as jnp
from tqdm import tqdm
import jax.scipy.stats as jstat
from jax import jit
import matplotlib.pyplot as plt
from tensorflow_probability.substrates import jax as tfp
import optax
tfpd = tfp.distributions

global GLOB_prox

from jax.config import config
config.update("jax_enable_x64", True)

# Go from a design matrix to the design mat with interaction and quadratic terms.
@jit
def expand_X(Xs, ind1_exp, ind2_exp):
    Xis = Xs[:, ind1_exp] * Xs[:, ind2_exp]
    Xqs = Xs*Xs
    #Xqs = Xqs[:, :-n_re]  # Remove quadratic terms for indicator functions
    Xhs = jnp.concatenate([Xs, Xis, Xqs], axis=1)
    return Xhs

@jit
def get_sd_svrg(grad_nll, grad_nll0, g0, grad_prior, mb_size, N):
    sd = {}
    for v in grad_nll:
        sd[v] = -(N/mb_size*(grad_nll[v] - grad_nll0[v]) + g0[v]  + grad_prior[v])
    return sd

#vv = mod.vv
#lr = ssi
@jit
def get_vv_at(vv, sd, ss, lr, tau0):
    # Gradient Step
    new_vv = {}
    for v in vv:
        new_vv[v] = vv[v] + lr*ss[v]*sd[v]

    # Proximal step
    #new_beta, new_lam = jax.lax.cond(invlam, jax_apply_prox_inv, jax_apply_prox)
    if GLOB_prox=='inv':
        new_beta, new_lam = jax_apply_prox_inv(new_vv['beta'], new_vv['lam'], tau0*lr*ss['beta'], tau0*lr*ss['lam'])
    elif GLOB_prox=='log':
        new_beta, new_lam = jax_apply_prox_log(new_vv['beta'], new_vv['lam'], tau0*lr*ss['beta'], tau0*lr*ss['lam'], 1/tau0)
    elif GLOB_prox=='std':
        new_beta, new_lam = jax_apply_prox(new_vv['beta'], new_vv['lam'], tau0*lr*ss['beta'], tau0*lr*ss['lam'])
    new_vv['beta'] = new_beta
    new_vv['lam'] = new_lam

    return new_vv

@jit
def get_sd(grad, ss):
    sd = {}
    for v in grad:
        sd[v] = -grad[v]*ss[v]
    return(sd)

class jax_vlMAP:
    auto_N_sg = 5000
    #auto_N_sg = 20000

    def __init__(self, X, y, lam_prior, lam_prior_vars, big_N = None, tau0=1, lik='normal', mb_size = None, no_adapt = False, invlam = False, track = False, lam_accel = True, l2_coef = 0., dont_pen = None, do_jit = True, logprox = False, beta0_init = None, quad = False, N_es = 1000, es_patience = 500):
        self.lik = lik
        self.tau0 = tau0
        self.big_N = big_N
        assert tau0 > 0
        self.no_adapt = no_adapt
        self.lam_prior_vars = lam_prior_vars 
        self.lam_prior = lam_prior
        self.invlam = invlam
        self.logprox = logprox
        self.track = track
        self.lam_accel = lam_accel
        self.l2_coef = l2_coef
        self.beta0_init = beta0_init 
        self.do_jit = do_jit

        self.quad = quad
        if self.quad:
            Pu = X.shape[1]
            self.ind1_exp = jnp.repeat(jnp.arange(Pu-1), 1+jnp.flip(jnp.arange(Pu-1)))
            self.ind2_exp = jnp.concatenate([jnp.arange(p+1, Pu) for p in range(Pu)])

        self.N_es = min(N_es, X.shape[0]//2)
        self.es_patience = es_patience # in iterations
        self.set_Xy(X, y)

        #assert not self.quad

        # Stochastic gradient settings.
        if mb_size is None:
            if self.N >= jax_vlMAP.auto_N_sg:
                self.mb_size = jax_vlMAP.auto_N_sg
            else:
                self.mb_size = self.N
        else:
            self.mb_size = mb_size
        if self.mb_size >= self.N:
            self.mb_size = self.N


        self.is_stochastic = self.mb_size < self.N

        #self.is_stochastic = True
        self.batches_per_epoch = int(np.ceil(self.N / self.mb_size))  # This is about ratio of dataset to mb_size.

        self.svrg_every = self.batches_per_epoch
        self.es_every = self.batches_per_epoch


        assert not (self.logprox and self.invlam)

        global GLOB_prox
        if self.invlam:
            GLOB_prox='inv'
        elif self.logprox:
            GLOB_prox='log'
        else:
            GLOB_prox='std'

        self.dont_pen = dont_pen
        if self.dont_pen is not None:
            self.do_pen = np.array(list(set(np.arange(self.Pb)).difference(dont_pen))).astype(int)
        else:
            self.do_pen = np.arange(self.Pb)

        #TODO: Verify still using.
        # Params
        self.ns_adam_pre = False
        self.tau_exp = True 
        self.omega_exp = True 
        self.sigma2_exp = True

        self.lam_prior_vars_init = {}
        for v in self.lam_prior_vars:
            self.lam_prior_vars_init[v] = jnp.copy(self.lam_prior_vars[v])

        self.reset_vars()
        self._compile_costs_updates(do_jit = self.do_jit)

        # Adam parameters.
        self.beta_adam = 0.5
        #self.beta_adam = 0.999
        self.eps_adam = 1e-7

        if self.lik in ['poisson','nb']:
            nbig = 5
            self.biginds = np.argpartition(self.y,-nbig)[-nbig:]
            self.hoardlike = True
        else:
            self.hoardlike = False

    def set_tau0(self, tau0):
        self.tau0 = tau0
        #self._compile_costs_updates(do_jit = self.do_jit)

    def set_Xy(self, X, y):
        N,self.Pu = X.shape

        ind_es = np.random.choice(N,self.N_es,replace=False)
        ind_train = np.setdiff1d(np.arange(N), ind_es)
        X_es = X[ind_es,:]
        if self.quad:
            X_es = expand_X(X_es, self.ind1_exp, self.ind2_exp)  # Create interaction terms just in time.
        y_es = y[ind_es]
        X_train = X[ind_train,:]
        y_train = y[ind_train]

        self.N = X_train.shape[0]

        if self.quad:
            self.P = 2*self.Pu + int(scipy.special.binom(self.Pu, 2))
        else:
            self.P = self.Pu

        if self.lik=='zinb':
            self.Pb=2*self.P
        else:
            self.Pb = self.P

        self.X = np.array(X_train)
        self.y = np.array(y_train)
        self.X_es = np.array(X_es)
        self.y_es = np.array(y_es)

    def reset_vars(self):
        # NOTE: This is indeed X.shape[0], NOT N. (Oh, is it still tho?)
        if self.X.shape[0] > 0:
            if self.lik in ['poisson','nb']:
                #beta0 = jnp.mean(jnp.log(self.y+0.5))
                beta0 = jnp.log(jnp.mean(self.y))
            elif self.lik=='zinb':
                iz = np.array(self.y)==0
                beta00 = jnp.mean(iz.astype(np.float64))
                beta0 = jnp.mean(jnp.log(self.y[~iz]+0.5))
            elif self.lik=='cauchy':
                beta0 = jnp.median(self.y)
            else:
                beta0 = jnp.mean(np.array(self.y).astype(np.float64))

            if self.lik in ['poisson','bernoulli','nb']:
                sigma2 = jnp.array(1.)
            elif self.lik in ['zinb']:
                sigma2 = jnp.array(1.)
            elif self.lik =='cauchy':
                mu_hat = jnp.median(self.y)
                sigma2 = jnp.median(jnp.abs(self.y - mu_hat))
            else:
                sigma2 = jnp.var(self.y)
        else:
            if self.beta0_init is None:
                beta0 = jnp.array(0.)
            else:
                beta0 = jnp.array(self.beta0_init)
            sigma2 = jnp.array(1.)
            if self.lik=='zinb':
                beta00 = jnp.array(0.)

        if self.sigma2_exp:
            sigma2 = jnp.log(sigma2)

        #PP = 2*self.P if self.lik=='zinb' else self.P
        beta = jnp.zeros(self.Pb)
        lam = 1.1*jnp.ones_like(beta)
        print("new lam init")
        #lam = jnp.ones_like(beta) / (self.N+1)

        if self.dont_pen is not None:
            lam = lam.at[self.dont_pen].set(0.) 

        self.vv = {
                'beta':beta,
                'lam':lam,
                'beta0':beta0,
                'sigma2':sigma2}
        if self.lik=='zinb':
            self.vv['beta00']=beta00
        #self.vv = self.vv | self.lam_prior_vars
        for v in self.lam_prior_vars:
            self.lam_prior_vars[v] = jnp.copy(self.lam_prior_vars_init[v])
        self.vv = {**self.vv, **self.lam_prior_vars}

        for v in self.vv:
            if len(self.vv[v].shape)==0:
                self.vv[v] = self.vv[v].reshape([1])
        self.last_beta = np.array(self.vv['beta'])

    def _compile_costs_updates(self, do_jit = True):

        def eval_nll_subset(vv, X, y):
            nll = -jnp.sum(self._predictive(X, vv).log_prob(y))
            return nll

        if GLOB_prox=='std':
            def eval_prior_nonsmooth(vv, tau0):
                ll = vv['lam']
                npd_ns = tau0*jnp.sum(ll * jnp.abs(vv['beta']))
                return npd_ns
        elif GLOB_prox=='log':
            def eval_prior_nonsmooth(vv, tau0):
                ll = vv['lam']
                npd_ns = tau0*jnp.sum(ll * jnp.abs(vv['beta']))
                llp_uni = -jnp.sum(jnp.log(vv['lam'][self.do_pen]))
                return npd_ns + llp_uni
        elif GLOB_prox=='inv':
            def eval_prior_nonsmooth(vv, tau0):
                ll = 1/vv['lam']
                npd_ns = tau0*jnp.sum(ll * jnp.abs(vv['beta']))
                return npd_ns
        else:
            assert False

        def eval_prior(vv, tau0):
            # Beta prior
            llp_uni = -jnp.sum(jnp.log(vv['lam'][self.do_pen]))
            if self.logprox:
                lam_lap = 0.
            else:
                lam_lap = llp_uni
            if self.invlam:
                lam_lap *= -1

            beta_l2 = self.l2_coef*self.N*jnp.sum(jnp.square(vv['beta']))/2
            
            # Lam prior
            lprior_lam = self.lam_prior(vv['lam'], vv, self)

            # sigma2 "prior"
            if not self.lik in ['poisson','bernoulli']:
                if self.sigma2_exp:
                    lprior_sigma2 = jnp.array(0) # The Reference prior for OLS?
                else:
                    lprior_sigma2 = jnp.log(vv['sigma2']) # The Reference prior for OLS
            else:
                lprior_sigma2 = jnp.array(0)
            prior_smooth = lam_lap + lprior_lam + lprior_sigma2 + beta_l2

            npd_ns = eval_prior_nonsmooth(vv, tau0)
            
            return prior_smooth, npd_ns

        def adam_update_ss_nojit(v_adam, vhat_adam, grad, vv, it):
                #grad = gradlike
            for v in v_adam:
                v_adam[v] = self.beta_adam*v_adam[v] + (1-self.beta_adam)*jnp.square(grad[v])

            for v in v_adam:
                vhat_adam[v] = v_adam[v] / (1-jnp.power(self.beta_adam,it+1))

            ss = {}
            for v in vv:
                ss[v] = 1/jnp.sqrt(vhat_adam[v]+self.eps_adam)

            return v_adam, vhat_adam, ss

        eval_nll_grad_subset = jax.value_and_grad(eval_nll_subset)
        eval_prior_grad = jax.value_and_grad(eval_prior, has_aux = True)

        def grad_samp(vv, tau0, Xs_use, ys):# Just used in one place below \shrug
            cost_nll, grad_nll = eval_nll_grad_subset(vv, Xs_use, ys)
            cost_npd, grad_npd = eval_prior_grad(vv, tau0)
            #cur_cost = cost_nll / len(samp) + sum(cost_npd) / self.N
            cur_cost = (self.N / Xs_use.shape[0]) * cost_nll  + sum(cost_npd)
            
            grad = {}
            for v in vv:
                grad[v] = (self.N / Xs_use.shape[0])* grad_nll[v] + grad_npd[v]
            return cur_cost, grad, grad_nll, grad_npd

        if do_jit:
            ## NOTE: If moving this, make sure to fix tau0 and any other globals.
            self.eval_nll_grad_subset = jit(eval_nll_grad_subset)
            self.eval_prior_grad = jit(eval_prior_grad)
            self.adam_update_ss = jit(adam_update_ss_nojit)
            self.grad_samp = jit(grad_samp)
        else:
            self.eval_nll_grad_subset = jax.value_and_grad(eval_nll_subset)
            self.eval_prior_grad = jax.value_and_grad(eval_prior, has_aux = True)
            self.adam_update_ss = adam_update_ss_nojit
            self.grad_samp = grad_samp

    def init_adam(self):
        return dict([(v,np.zeros_like(self.vv[v])) for v in self.vv]), dict([(v,np.zeros_like(self.vv[v])) for v in self.vv])

    def fit(self, max_iters = 10000, lr_pre=1e-2, verbose = True, debug = np.inf, hoardlarge = False, ada = False, warm_up = False):
        global GLOB_prox
        if ada:
            lr = lr_pre
        else:
            lr = lr_pre / self.N

        es_num = int(np.ceil(max_iters / self.es_every))

        ss_ones = {}
        for v in self.vv:
            ss_ones[v] = np.ones_like(self.vv[v])

        if ada:
            self.v_adam = {}
            self.vhat_adam = {}
            for v in self.vv:
                self.v_adam[v] = np.zeros_like(self.vv[v])
                self.vhat_adam[v] = np.zeros_like(self.vv[v])


        self.costs = np.zeros(max_iters)*np.nan
        #self.sparsity = np.zeros(max_iters)*np.nan
        self.nll_es = np.zeros(es_num)*np.nan

        self.tracking = {}
        if self.track:
            for v in self.vv:
                self.tracking[v] = np.zeros((max_iters,)+self.vv[v].shape)

        for i in tqdm(range(max_iters), disable = not verbose, smoothing = 0.):

            if self.tracking:
                for v in self.vv:
                    self.tracking[v][i] = self.vv[v]

            if i % self.svrg_every==0:
                self.vv0 = {}
                for v in self.vv:
                    self.vv0[v] = jnp.copy(self.vv[v])

                g0 = dict([(v, np.zeros_like(self.vv[v])) for v in self.vv])
                for iti in tqdm(range(self.batches_per_epoch), leave=False, disable = not verbose):
                    samp_vr = np.arange(iti*self.mb_size,np.minimum((iti+1)*self.mb_size,self.N))
                    Xs_vr = self.X[samp_vr, :]
                    ys_vr = self.y[samp_vr]
                    if self.quad:
                        X_use_vr = expand_X(Xs_vr, self.ind1_exp, self.ind2_exp)  # Create interaction terms just in time.
                    else:
                        X_use_vr = Xs_vr  # Create interaction terms just in time.
                    _, grad_vr = self.eval_nll_grad_subset(self.vv0, X_use_vr, ys_vr)
                    for v in self.vv:
                        g0[v] += grad_vr[v]

            if i % self.es_every==0:
                self.nll_es[i//self.es_every] = -np.sum(self.predictive(self.X_es).log_prob(self.y_es))
                best_it = np.nanargmin(self.nll_es) * self.es_every
                if i-best_it > self.es_patience:
                    print("ES stop!")
                    break

            ind = np.random.choice(self.N,self.mb_size,replace=False)

            # TODO: Expand quadratic boy here.
            Xs = self.X[ind,:]
            ys = self.y[ind]
            if self.quad:
                Xs_use = expand_X(Xs, self.ind1_exp, self.ind2_exp)  # Create interaction terms just in time.
            else:
                Xs_use = Xs
            cost_nll, grad_nll = self.eval_nll_grad_subset(self.vv, Xs_use, ys)
            _, grad_nll0 = self.eval_nll_grad_subset(self.vv0, Xs_use, ys)
            cost_prior, grad_prior = self.eval_prior_grad(self.vv, self.tau0)

            self.costs[i] = self.N/self.mb_size*cost_nll + sum(cost_prior)
            #self.sparsity[i] = np.mean(self.vv['beta']==0)
            if not np.isfinite(self.costs[i]):
                print("Infinite cost!")
                import IPython; IPython.embed()
                break

            sd = get_sd_svrg(grad_nll, grad_nll0, g0, grad_prior, self.mb_size, self.N)
            if ada:
                #self.v_adam, self.vhat_adam, ss_adam = self.adam_update_ss(self.v_adam, self.vhat_adam, sd, self.vv, i)
                #ss = ss_adam

                #ss = {}
                #for v in self.vv:
                #    if v in self.lam_prior_vars:
                #        ss[v] = ss_adam[v]
                #    else:
                #        ss[v] = ss_ones[v]/N
                #ss['lam'] = ss_ones['lam']*0.1
                #ss['lam'] = jnp.minimum(ss_ones['lam'], ss['lam'])
                adam_grad = {}
                for v in self.vv:
                    adam_grad[v] = self.N/self.mb_size*grad_nll[v] + grad_prior[v]
                self.v_adam, self.vhat_adam, ss = self.adam_update_ss(self.v_adam, self.vhat_adam, adam_grad, self.vv, i)
            else:
                ss = ss_ones
            wu_its = 5
            if warm_up and i < wu_its:
                wu_rates = np.logspace(-8,np.log10(lr),num=wu_its)
                lr_use = wu_rates[i]
            else:
                lr_use = lr
            next_vv = get_vv_at(self.vv, sd, ss, lr_use, self.tau0)
            #import IPython; IPython.embed()
            self.vv = next_vv
            self.last_it = i

        self.beta = self.vv['beta']

    def _predictive(self, XX, vv):
        if self.sigma2_exp:
            stddev = jnp.exp(vv['sigma2']/2)
        else:
            stddev = jnp.sqrt(vv['sigma2'])

        if self.lik == 'zinb':
            preds_y = XX @ vv['beta'][:self.P] + vv['beta0']
            preds_z = XX @ vv['beta'][self.P:] + vv['beta00']
        else:
            preds = XX @ vv['beta'] + vv['beta0']

        if self.lik == 'normal':
            dist_pred = tfpd.Normal(preds, scale=stddev)
        elif self.lik == 'cauchy':
            dist_pred = tfpd.Cauchy(loc=preds, scale=stddev)
        elif self.lik == 'bernoulli':
            dist_pred = tfpd.Bernoulli(logits=preds)
        elif self.lik == 'poisson':
            dist_pred = tfpd.Poisson(log_rate=preds)
        elif self.lik == 'nb':
            dist_pred = tfpd.NegativeBinomial(total_count=1/jnp.square(stddev), logits=preds)
        elif self.lik=='zinb':
            dist_nz = tfpd.NegativeBinomial(total_count=1/jnp.square(stddev), logits = preds_y)
            dist_z = tfpd.Deterministic(loc=jnp.zeros_like(preds_y))
            pnz = tfpd.Categorical(logits=jnp.stack([preds_z,-preds_z]).T)
            dist_pred = tfpd.Mixture(pnz, components=[dist_z, dist_nz])
        else:
            raise Exception("Unrecognized Likelihood.")
        return dist_pred

    def predictive(self, XX):
        return self._predictive(XX, self.vv)

    def plot(self, fname = 'fit.png'):
        if self.track:
            nvars = len(self.tracking)
        else:
            nvars = 0
        nfigs = nvars + 3
        if self.is_stochastic:
            nfigs += 1
        nrows = int(np.ceil(nfigs/2))
        ncols = 2

        #fig = plt.figure(figsize=[ncols*3,nrows*1.5])
        fig = plt.figure(figsize=[ncols*5,nrows*2])

        plt.subplot(nrows,ncols,1)
        plt.plot(self.costs[:(self.last_it+1)])
        plt.title('Cost')

        #plt.subplot(nrows,ncols,2)
        ##plt.plot(self.step_sizes[:self.last_it], label = 'step size')
        ##plt.plot(self.reltols[:self.last_it], label = 'reltol')
        ##plt.plot(self.beta_dinf[:self.last_it], label = r"$\beta$ dinf")
        #plt.legend(prop={'size':5})
        #plt.yscale('log')
        #plt.title('Convergence')

        ##if self.is_stochastic:
        #    plt.subplot(nrows,ncols,3)
        #    plt.plot(self.grad_norm[:self.last_it], label = 'Stoch Grad Norm')
        #    plt.plot(self.grad_norm_vr[:self.last_it], label = 'VR Grad Norm')
        #    plt.legend(prop={'size':5})
        #    plt.yscale('log')
        #    plt.title('Gradient Norms')
        #else:
        plt.subplot(nrows,ncols,3)
        plt.plot(np.log10(1+self.costs[:self.last_it]-np.min(self.costs[:self.last_it])))
        plt.title('Optimality Gap (log(1+))')

        if self.track:
            for vi,v in enumerate(self.tracking):
                plt.subplot(nrows,ncols,1+3+vi)
                plt.plot(self.tracking[v][:(self.last_it+1)])
                if v=='beta':
                    sparsity = np.mean(self.tracking[v][:self.last_it]==0,axis=1)
                    ax = plt.gca().twinx()
                    ax.plot(sparsity, color = 'green', linestyle='--')
                if v=='lam':
                    plt.yscale('log')
                plt.title(v)

        plt.tight_layout()
        plt.savefig(fname)
        plt.close()

#@jit
def jax_prox_inv_cost(x,lam,x0,lam0,sx,sl):
    return jnp.abs(x)/lam + jnp.square(x-x0)/(2*sx) + jnp.square(lam-lam0)/(2*sl)


## Can't jit because of the rootfinding.
def jax_apply_prox_inv(x0, lam0, sx, sl):
    p = jnp.stack([1/(sx*sl), -lam0/(sx*sl), jnp.zeros(P), -jnp.abs(x0)/sx, jnp.ones(P)]).T

    roots = quartic_roots(p)

    lamcand1 = roots[:,2].real
    xcand1 = jnp.maximum(0,jnp.abs(x0)-sx*lamcand1)*jnp.sign(x0)
    cost1 = jax_prox_inv_cost(xcand1, lamcand1, x0, lam0, sx, sl)
    lamcand2 = roots[:,3].real
    xcand2 = jnp.maximum(0,jnp.abs(x0)-sx*lamcand2)*jnp.sign(x0)
    cost2 = jax_prox_inv_cost(xcand2, lamcand2, x0, lam0, sx, sl)

    # TODO: Comparison seems unnecessary; always seems to be upper root (lower root seems to be inflection point).
    c1w = cost2>cost1
    lamcand = c1w*lamcand1 + (1-c1w)*lamcand2
    xcand = c1w*xcand1 + (1-c1w)*xcand2
    xiszero = (lam0*jnp.abs(x0))/sx < 1
    lamcand = xiszero*lam0 + (1-xiszero)*lamcand
    xcand *= (1-xiszero)*xcand

    return xcand, lamcand

def sto(x0,lam,sx):
    return jnp.maximum(0,jnp.abs(x0)-lam*sx)*jnp.sign(x0)

#@jit
def jax_apply_prox(x0, lam0, sx, sl):
    is_convex = sl*sx<1

    nc_sw = (lam0/sl) < (jnp.abs(x0)/sx)
    nonconv_sol = (1-nc_sw) * lam0 # + nc_sw*0

    c_sw = lam0<jnp.abs(x0)/sx
    rect_lam = jnp.maximum(0,lam0)
    conv_sol = jnp.minimum(jnp.maximum(0,lam0-sl*jnp.abs(x0)) / (1-sl*sx), rect_lam)

    lam_sol = jnp.where(is_convex, conv_sol, nonconv_sol)
    # TODO: use STO.
    x_sol = jnp.maximum(0,jnp.abs(x0)-lam_sol*sx)*jnp.sign(x0)

    return x_sol, lam_sol

def jax_prox_log_cost(x,lam,x0,lam0,sx,sl,c):
    return jnp.abs(x)*lam + jnp.square(x-x0)/(2*sx) + jnp.square(lam-lam0)/(2*sl) - c*jnp.log(lam)

#ss = 100
#x0 = np.random.normal(size=ss)
#lam0 = np.abs(np.random.normal(size=ss))
#sx = np.abs(np.random.normal(size=ss))
#sl = np.abs(np.random.normal(size=ss))
#c = np.abs(np.random.normal(size=ss))

## Have to check zero first for this example:
#x0 = -0.3012135452028746
#lam0 = 1.5214325031249438
#sx = 0.7177978263375587
#sl = 1.967460846559591
##a = -0.5922949078923144
#c = 0.5922949078923144
# Warning: this function will produce garbage if c < 0.

#its = 100
#for it in range(its):
#    x0 = np.random.normal()
#    lam0 = np.abs(np.random.normal())
#    sx = np.abs(np.random.normal())
#    sl = np.abs(np.random.normal())
#    c = np.abs(np.random.normal())
#    #x0 = 1.2247781273189027
#    #lam0 = 0.7833907802663115
#    #sx = 1.385556945147132
#    #sl = 2.360741208029606
#    #c = 0.06447148525587204
#    #x0 = 1.2
#    #lam0 = 0.8
#    #sx = 1.4
#    #sl = 2.3
#    #c = 0.05
def jax_apply_prox_log(x0, lam0, sx, sl, c):
    ## Check if x is going to be zero.
    rootdisc_iz = jnp.sqrt(jnp.square(lam0) + 4 * c * sl)
    lamizm = jnp.maximum(0,(lam0 - rootdisc_iz) / 2)
    lamizp = jnp.maximum(0,(lam0 + rootdisc_iz) / 2)

    costm = jax_prox_log_cost(0., lamizm, x0, lam0, sx, sl, c)
    costp = jax_prox_log_cost(0., lamizp, x0, lam0, sx, sl, c)
    mwins = costm<costp
    lamiz = jnp.where(mwins, lamizm, lamizp)

    xiz = sto(x0,lamiz,sx)
    zerocost = jnp.where(mwins, costm, costp)

    ## Get value if x is nonzero
    qa = sx-1/sl
    qb = lam0/sl - jnp.abs(x0)
    qc = c # Coefficient for -log(lam).
    disc = jnp.square(qb)-4*qa*qc
    discroot = jnp.sqrt(jnp.maximum(0,disc)) # Threshold negative to zero, anticipate inferior cost will weed it out.
    lm = jnp.maximum(0,(-qb - discroot) / (2*qa))
    lp = jnp.maximum(0,(-qb + discroot) / (2*qa))

    xm = sto(x0,lm,sx)
    xp = sto(x0,lp,sx)

    costm = jax_prox_log_cost(xm, lm, x0, lam0, sx, sl, c)
    costp = jax_prox_log_cost(xp, lp, x0, lam0, sx, sl, c)
    mwins = costm<costp
    x_sol = jnp.where(mwins, xm, xp)
    lam_sol = jnp.where(mwins, lm, lp)
    nzcost = jnp.where(mwins, costm, costp)

    ## Check if we return zero or nonzero value.
    xiszero = zerocost < nzcost
    x_sol = jnp.where(xiszero, xiz, x_sol)
    lam_sol = jnp.where(xiszero, lamiz, lam_sol)

    return x_sol, lam_sol
