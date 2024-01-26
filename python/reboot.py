#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/reboot.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.26.2024

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

assert sparsity_type=='random'
mod = jax_vlMAP(X, y, adaptive_prior, {}, logprox = True)

iters = 10000
#iters = 300

#lr = 5e-1/N
#tau0 = 0.2 * N
tau0 = 0.1 * N
prop_es = 0.1
#mb_size = 256
mb_size = 1024
#mb_size = N
lr = 5e-2 / N
#lr = 5e-3 / mb_size

ss = {}
for v in mod.vv:
    ss[v] = np.ones_like(mod.vv[v])

@jit
def get_sd2(grad_nll, grad_prior, ss, mb_size, N):
    sd = {}
    for v in grad_nll:
        sd[v] = -(N/mb_size*grad_nll[v] + grad_prior[v])
    return sd

costs = np.zeros(iters)*np.nan
for i in tqdm(range(iters)):
    ind = np.random.choice(N,mb_size,replace=False)

    cost_nll, grad_nll = mod.eval_nll_grad_subset(mod.vv, X[ind,:], y[ind])
    cost_prior, grad_prior = mod.eval_prior_grad(mod.vv, tau0)

    costs[i] = N/mb_size*cost_nll + sum(cost_prior)
    if not np.isfinite(costs[i]):
        print("Infinite cost!")
        break

    #sd = get_sd(grad, ss)
    sd = get_sd2(grad_nll, grad_prior, ss, mb_size, N)
    next_vv = get_vv_at(mod.vv, sd, ss, lr, tau0)
    mod.vv = next_vv

fig = plt.figure()
plt.plot(costs)
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
