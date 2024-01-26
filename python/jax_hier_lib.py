#def adaptive_prior(x, pv):
#    tau = jnp.exp(pv['log_tau'])
#    tau_dens = -jstat.norm.logpdf(pv['log_tau'])
#    #lam_dist = tfpd.Cauchy(loc=N, scale=tau)
#    lam_dist = tfpd.Cauchy(loc=N*jnp.exp(-pv['sigma2']), scale=tau) # TODO: Figure out how to involve sigma2.
#    lam_dens = -jnp.sum(lam_dist.log_prob(x)-jnp.log((1-lam_dist.cdf(0))))
#    return tau_dens + lam_dens

#@jit
#def adaptive_prior(x, pv):
#    lam_dist = tfpd.Cauchy(loc=N*jnp.exp(-pv['sigma2']), scale=1.) # TODO: Figure out how to involve sigma2.
#    lam_dens = -jnp.sum(lam_dist.log_prob(x)-jnp.log((1-lam_dist.cdf(0))))
#    return lam_dens

## TODO: N,P is global.
#def adaptive_prior(x, pv):
#    #lam_dist = tfpd.Cauchy(loc=N*jnp.exp(-pv['sigma2']), scale=1.) 
#    lam_dist = tfpd.Cauchy(loc=N*jnp.exp(-pv['sigma2']), scale=P*jnp.exp(pv['sigma2']))
#    lam_dens = -jnp.sum(lam_dist.log_prob(x)-jnp.log((1-lam_dist.cdf(0))))
#    return lam_dens


def adaptive_prior(x, pv, mod):
    #lam_dist = tfpd.Cauchy(loc=self.N*jnp.exp(-pv['sigma2']), scale=1.) 
    lam_dist = tfpd.Cauchy(loc=0., scale=1.)
    lam_dens = -jnp.sum(lam_dist.log_prob(x)-jnp.log((1-lam_dist.cdf(0))))
    return lam_dens

def group_prior(x, pv, mod):
    gamma = jnp.exp(pv['log_gamma'])
    #print("Is it someone new.")
    #gamma = jnp.maximum(0., pv['log_gamma'])
    gamma_dist = tfpd.Cauchy(loc=0., scale=1.)
    gamma_dens = -jnp.sum(gamma_dist.log_prob(gamma)-jnp.log((1-gamma_dist.cdf(0))))
    #gamma_dens = 0.

    gamma_big = gamma[groups]
    lam_sd = 1./jnp.sqrt(mod.N)
    #lam_sd = 1./mod.N
    lam_dist = tfpd.Normal(loc=gamma_big, scale=lam_sd)
    lam_dens = -jnp.sum(lam_dist.log_prob(x)-jnp.log(1-lam_dist.cdf(0)))
    return lam_dens + gamma_dens

def hier_prior(x, pv, mod):
    gamma = jnp.exp(pv['log_gamma'])
    gamma_dist = tfpd.Cauchy(loc=0., scale=1.)
    gamma_dens = -jnp.sum(gamma_dist.log_prob(gamma)-jnp.log((1-gamma_dist.cdf(0))))

    #lam_sd = 1.
    #lam_sd = 1e-2
    #lam_sd = 1e-1
    lam_sd = 1/jnp.sqrt(mod.N)
    #lam_sd = 1/mod.N

    #gam_meq = jnp.apply_along_axis(lambda x: jnp.exp(jnp.mean(jnp.log(x))), 0, gamma[GAMMAt])

    GAMMAt = make_gamma_mat(Pu).astype(int)
    #gam_meq = jnp.min(gamma[GAMMAt], axis = 0)
    #gam_meq = jnp.mean(gamma[GAMMAt], axis = 0)
    #jnp.sum(jax.nn.softmax(-gamma[GAMMAt][:,i]/temp) * gamma[GAMMAt][:,i])
    temp = 1/jnp.sqrt(Pu)
    gam_meq = jnp.apply_along_axis(lambda x: jnp.sum(jax.nn.softmax(-x/temp) * x), 0, gamma[GAMMAt])
    meq_dist =  tfpd.Normal(loc=gam_meq, scale=lam_sd)

    lam_me = x[np.arange(Pu)]#tf.gather(lam, np.arange(Pu))
    lam_q = x[np.arange(Pu+Pi,Pu+Pi+Pq)]
    me_dens = -jnp.sum(meq_dist.log_prob(lam_me)-jnp.log(1-meq_dist.cdf(0)))
    q_dens = -jnp.sum(meq_dist.log_prob(lam_q)-jnp.log(1-meq_dist.cdf(0)))

    lam_i = x[np.arange(Pu,Pu+Pi)]
    i_dist =  tfpd.Normal(loc=gamma, scale=lam_sd)
    i_dens = -jnp.sum(i_dist.log_prob(lam_i)-jnp.log(1-i_dist.cdf(0)))

    return gamma_dens + me_dens + q_dens + i_dens

#def hier_prior(x, pv, mod):
#    gamma = jnp.exp(pv['log_gamma'])
#    gamma_dist = tfpd.Cauchy(loc=0., scale=1.)
#    gamma_dens = -jnp.sum(gamma_dist.log_prob(gamma)-jnp.log((1-gamma_dist.cdf(0))))
#
#    #lam_sd = 1.
#    #lam_sd = 1e-2
#    #lam_sd = 1e-1
#    lam_sd = 1/jnp.sqrt(mod.N)
#
#    GAMMAt = make_gamma_mat(Pu).astype(int)
#    #gam_meq = jnp.min(gamma[GAMMAt], axis = 0)
#    #gam_meq = jnp.apply_along_axis(lambda x: jnp.exp(jnp.mean(jnp.log(x))), 0, gamma[GAMMAt])
#    meq_dist =  tfpd.Normal(loc=gamma[GAMMAt], scale=lam_sd)
#
#    lam_me = x[np.arange(Pu)]#tf.gather(lam, np.arange(Pu))
#    lam_q = x[np.arange(Pu+Pi,Pu+Pi+Pq)]
#    lam_me = jnp.tile(lam_me, [Pu-1,1])
#    lam_q = jnp.tile(lam_q, [Pu-1,1])
#    me_dens = -jnp.sum(meq_dist.log_prob(lam_me)-jnp.log(1-meq_dist.cdf(0))) #- jnp.log(Pu-1)
#    q_dens = -jnp.sum(meq_dist.log_prob(lam_q)-jnp.log(1-meq_dist.cdf(0))) #- jnp.log(Pu-1)
#
#    lam_i = x[np.arange(Pu,Pu+Pi)]
#    i_dist =  tfpd.Normal(loc=gamma, scale=lam_sd)
#    i_dens = -jnp.sum(i_dist.log_prob(lam_i)-jnp.log(1-i_dist.cdf(0)))
#
#    return gamma_dens + me_dens + q_dens + i_dens

""" def group_prior(x, pv, mod):
    gamma = jnp.exp(pv['log_gamma'])
    gamma_dist = tfpd.Cauchy(loc=mod.N*jnp.exp(-pv['sigma2']), scale=mod.P*jnp.exp(pv['sigma2']))
    gamma_dens = -jnp.sum(gamma_dist.log_prob(gamma)-jnp.log((1-gamma_dist.cdf(0))))

    raise Exception
    gamma_big = gamma[groups]
    lam_dist = tfpd.TruncatedNormal(loc=gamma_big, scale=1., low = 0., high = 1e10) # TODO: Include sigma2, N, P here?
    lam_dens = -jnp.sum(lam_dist.log_prob(x))
    return lam_dens + gamma_dens """

#def group_prior(x, pv, mod):
#    gamma = jnp.exp(pv['log_gamma'])
#    gamma_dist = tfpd.Cauchy(loc=mod.N*jnp.exp(-pv['sigma2']), scale=mod.P*jnp.exp(pv['sigma2']))
#    gamma_dens = -jnp.sum(gamma_dist.log_prob(gamma)-jnp.log((1-gamma_dist.cdf(0))))
#
#    gamma_big = gamma[groups]
#    lam_sd = 1.
#    lam_dist = tfpd.Normal(loc=gamma_big, scale=lam_sd)
#    lam_dens = -jnp.sum(lam_dist.log_prob(x)-jnp.log(1-lam_dist.cdf(0)))
#    return lam_dens + gamma_dens
#
#def hier_prior(x, pv, mod):
#    gamma = jnp.exp(pv['log_gamma'])
#    gamma_dist = tfpd.Cauchy(loc=mod.N*jnp.exp(-pv['sigma2']), scale=mod.P*jnp.exp(pv['sigma2']))
#    gamma_dens = -jnp.sum(gamma_dist.log_prob(gamma)-jnp.log((1-gamma_dist.cdf(0))))
#
#    #lam_sd = 1.
#    lam_sd = 1e-2
#
#    GAMMAt = make_gamma_mat(Pu).astype(int)
#    gam_meq = jnp.min(gamma[GAMMAt], axis = 0)
#    meq_dist =  tfpd.Normal(loc=gam_meq, scale=lam_sd)
#    #meq_dens = -jnp.sum(lam_dist.log_prob(x)-jnp.log(1-lam_dist.cdf(0)))
#
#    lam_me = x[np.arange(Pu)]#tf.gather(lam, np.arange(Pu))
#    lam_q = x[np.arange(Pu+Pi,Pu+Pi+Pq)]
#    me_dens = -jnp.sum(meq_dist.log_prob(lam_me)-jnp.log(1-meq_dist.cdf(0)))
#    #me_dens = -meq_dist.log_prob(tf.math.log(lam_me))
#    q_dens = -jnp.sum(meq_dist.log_prob(lam_q)-jnp.log(1-meq_dist.cdf(0)))
#    #q_dens = -meq_dist.log_prob(tf.math.log(lam_q))
#
#    lam_i = x[np.arange(Pu,Pu+Pi)]
#    i_dist =  tfpd.Normal(loc=gamma, scale=lam_sd)
#    i_dens = -jnp.sum(i_dist.log_prob(lam_i)-jnp.log(1-i_dist.cdf(0)))
#
#    return gamma_dens + me_dens + q_dens + i_dens
#
#def hier_prior(x, pv, mod):
#    gamma = jnp.exp(pv['log_gamma'])
#    gamma_dist = tfpd.Cauchy(loc=mod.N*jnp.exp(-pv['sigma2']), scale=mod.P*jnp.exp(pv['sigma2']))
#    gamma_dens = -jnp.sum(gamma_dist.log_prob(gamma)-jnp.log((1-gamma_dist.cdf(0))))
#
#    #lam_sd = 1.
#    lam_sd = 1e-2
#
#    GAMMAt = make_gamma_mat(Pu).astype(int)
#    gam_meq = jnp.min(gamma[GAMMAt], axis = 0)
#    meq_dist =  tfpd.Normal(loc=gam_meq, scale=lam_sd)
#    #meq_dens = -jnp.sum(lam_dist.log_prob(x)-jnp.log(1-lam_dist.cdf(0)))
#
#    lam_me = x[np.arange(Pu)]#tf.gather(lam, np.arange(Pu))
#    lam_q = x[np.arange(Pu+Pi,Pu+Pi+Pq)]
#    me_dens = -jnp.sum(meq_dist.log_prob(lam_me)-jnp.log(1-meq_dist.cdf(0)))
#    #me_dens = -meq_dist.log_prob(tf.math.log(lam_me))
#    q_dens = -jnp.sum(meq_dist.log_prob(lam_q)-jnp.log(1-meq_dist.cdf(0)))
#    #q_dens = -meq_dist.log_prob(tf.math.log(lam_q))
#
#    lam_i = x[np.arange(Pu,Pu+Pi)]
#    i_dist =  tfpd.Normal(loc=gamma, scale=lam_sd)
#    i_dens = -jnp.sum(i_dist.log_prob(lam_i)-jnp.log(1-i_dist.cdf(0)))
#
#    return gamma_dens + me_dens + q_dens + i_dens
#
#
