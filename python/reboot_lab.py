#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/reboot_lab.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.26.2024

exec(open("python/jax_nsa.py").read())
exec(open("python/jax_hier_lib.py").read())
exec(open("python/sim_settings.py").read())
exec(open("python/sim_lib.py").read())


sigma = 1.

manual = True
np.random.seed(123)

if manual:
    for i in range(10):
        print("Manual")
    s_i = '0'
    seed = 123
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

    if manual:
        N = 10000
        Pu = 1000
        Pnz = 1

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
    if sparsity_type=='hiernd':
        Xu = Xu_all[:N,:]
    XX = X_all[N:(N+NN),:]
    y = y_all[:N]
    yy = y_all[N:(N+NN)]
elif sim == 'libsvm':
    with open("pickles/"+s_i+".pkl",'rb') as f:
        X_all, y_all = pickle.load(f)

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
    XX = X_all[test_set,:]
    XXu = XXu_all[test_set,:]
    yy = y_all[test_set]
    X = X_all[train_set,:]
    Xu = Xu_all[train_set,:]
    y = y_all[train_set]
    NN = len(test_set)
    N = len(train_set)
else:
    raise Exception("Dataset not recognized.")


iters = 10000

#prop_es = 0.1
N_es = 1000
mb_size = 256
es_patience = 500 # in iterations

ind_es = np.random.choice(N,N_es,replace=False)
ind_train = np.setdiff1d(np.arange(N), ind_es)
X_es = X[ind_es,:]
y_es = y[ind_es]
X_train = X[ind_train,:]
y_train = y[ind_train]

assert sparsity_type=='random'
mod = jax_vlMAP(X_train, y_train, adaptive_prior, {}, logprox = True, mb_size = mb_size)

lr = 5e-2 / mod.N
tau0 = 0.1 * mod.N

verbose = True

batches_per_epoch = int(np.ceil(mod.N / mod.mb_size))
assert batches_per_epoch==mod.batches_per_epoch
svrg_every = mod.batches_per_epoch
es_every = mod.batches_per_epoch
es_num = int(np.ceil(iters / es_every))

ss = {}
for v in mod.vv:
    ss[v] = np.ones_like(mod.vv[v])

@jit
def get_sd2(grad_nll, grad_prior, ss, mb_size, N):
    sd = {}
    for v in grad_nll:
        sd[v] = -(N/mb_size*grad_nll[v] + grad_prior[v])
    return sd

@jit
def get_sd_svrg(grad_nll, grad_nll0, g0, grad_prior, ss, mb_size, N):
    sd = {}
    for v in grad_nll:
        sd[v] = -(N/mb_size*(grad_nll[v] - grad_nll0[v]) + g0[v]  + grad_prior[v])
    return sd

costs = np.zeros(iters)*np.nan
nll_es = np.zeros(es_num)*np.nan
for i in tqdm(range(iters)):

    if i % svrg_every==0:
        mod.vv0 = {}
        for v in mod.vv:
            mod.vv0[v] = jnp.copy(mod.vv[v])

        g0 = dict([(v, np.zeros_like(mod.vv[v])) for v in mod.vv])
        for iti in tqdm(range(batches_per_epoch), leave=False, disable = not verbose):
            samp_vr = np.arange(iti*mod.mb_size,np.minimum((iti+1)*mod.mb_size,mod.N))
            Xs_vr = mod.X[samp_vr, :]
            ys_vr = mod.y[samp_vr]
            if mod.quad:
                X_use_vr = expand_X(Xs_vr, mod.ind1_exp, mod.ind2_exp)  # Create interaction terms just in time.
            else:
                X_use_vr = Xs_vr  # Create interaction terms just in time.
            _, grad_vr = mod.eval_nll_grad_subset(mod.vv0, X_use_vr, ys_vr)
            for v in mod.vv:
                g0[v] += grad_vr[v]

    if i % es_every==0:
        nll_es[i//es_every] = -np.sum(mod.predictive(X_es).log_prob(y_es))
        best_it = np.nanargmin(nll_es) * es_every
        if i-best_it > es_patience:
            print("ES stop!")
            break

    ind = np.random.choice(mod.N,mod.mb_size,replace=False)

    cost_nll, grad_nll = mod.eval_nll_grad_subset(mod.vv, mod.X[ind,:], mod.y[ind])
    _, grad_nll0 = mod.eval_nll_grad_subset(mod.vv0, mod.X[ind,:], mod.y[ind])
    cost_prior, grad_prior = mod.eval_prior_grad(mod.vv, tau0)

    costs[i] = mod.N/mod.mb_size*cost_nll + sum(cost_prior)
    if not np.isfinite(costs[i]):
        print("Infinite cost!")
        break

    #sd = get_sd(grad, ss)
    #sd = get_sd2(grad_nll, grad_prior, ss, mod.mb_size, N)
    sd = get_sd_svrg(grad_nll, grad_nll0, g0, grad_prior, ss, mod.mb_size, mod.N)
    next_vv = get_vv_at(mod.vv, sd, ss, lr, tau0)
    mod.vv = next_vv

fig = plt.figure()
plt.plot(costs)
ax = plt.gca()
ax1 = ax.twinx()
ax1.plot(es_every*np.arange(es_num), nll_es, color = 'orange')
plt.savefig("temp.pdf")
plt.close()


print(mod.vv['beta'][:10])

print("estimate locs:")
print(np.where(mod.vv['beta']!=0))
print("true locs:")
np.where(beta!=0)

#if manual:
#    for i in range(10):
#        print("Manual")


