
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/reboot_lab.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.26.2024

exec(open("python/jax_nsa.py").read())
exec(open("python/jax_hier_lib.py").read())
exec(open("python/sim_settings.py").read())
exec(open("python/sim_lib.py").read())

exec(open('python/glmnet_wrapper.py').read())
#exec(open('python/MLGL_wrapper_e11.py').read())
#exec(open('python/ida_load.py').read())


sigma = 1.

manual = True
np.random.seed(123)

if manual:
    for i in range(10):
        print("Manual")
    s_i = '0'
    #s_i = '1'
    #s_i = '2'
    #s_i = '3'
    #s_i = '4'
    #s_i = '5'
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
        #N = 100000
        Pu = 1000
        #Pu = 10000
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
mod = jax_vlMAP(X, y, adaptive_prior, {}, logprox = True, mb_size = 256, lik = lik)

tau0 = 0.1 * mod.N 

verbose = True

mod.fit(max_iters = max_iters)

#fig = plt.figure()
#plt.plot(costs)
#ax = plt.gca()
#ax1 = ax.twinx()
##ax1.plot(es_every*np.arange(es_num), nll_es, color = 'orange')
#eps = 1e-8
#ax1.plot(es_every*np.arange(es_num), np.log10(nll_es+eps-np.nanmin(nll_es)), color = 'orange')
##ax1.plot(sparsity, color = 'orange')
#plt.savefig("temp.pdf")
#plt.close()

print(mod.vv['beta'][:10])

print("estimate")
print(np.where(mod.vv['beta']!=0))
print(mod.vv['beta'][mod.vv['beta']!=0])
print("true locs:")
print(beta[beta!=0])

#if manual:
#    for i in range(10):
#        print("Manual")


tt = time()
res = glmnet_fit(X, y, XX, lik, taus = tau0/N)
td = time() - tt
print(td)
