#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  mosaic_nn_lib.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 04.12.2024

analysis = 'lowpenalty'
#analysis = 'highpenalty'

if analysis == 'lowpenalty':
    step_size = 1e-3
    #tau0 = 1e1
    #tau0 = 0.
elif analysis == 'highpenalty':
    step_size = 2.5e-4
    tau0 = 1e2
else:
   raise Exception("Analysis arg should be either lowpenalty or highpenalty")

seed = 0
lam_eventual_step = step_size
lam_init_step = 0.
lam_init = 1e-8
num_epochs = 10000
lam1at = num_epochs//2

# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = jax.random.split(key)
  return scale * jax.random.normal(w_key, (n, m)), scale * jax.random.normal(b_key, (n,))

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
  keys = jax.random.split(key, len(sizes))
  return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


def relu(x):
  return jnp.maximum(0, x)

def predict(params, image):
  # per-example predictions
  activations = image
  w,b = params[0]
  activations = w @ activations# + b[:,jnp.newaxis]
  for w, b in params[1:-1]:
    #outputs = jnp.dot(w, activations) + b
    outputs = w @ activations + b[:,jnp.newaxis]
    activations = relu(outputs)
  
  final_w, final_b = params[-1]
  logits = final_w @ activations + final_b[:,jnp.newaxis]
  return logits
  #return logits - logsumexp(logits)

def nn_prior(x, log_gamma, X):
    N,P = X.shape

    gamma = jnp.exp(log_gamma)
    gamma_dist = tfpd.Cauchy(loc=N, scale=P)
    gamma_dens = -jnp.sum(gamma_dist.log_prob(gamma)-jnp.log((1-gamma_dist.cdf(0))))

    gamma_big = jnp.tile(gamma,[2,1])
    lam_sd = 1.
    lam_dist = tfpd.Normal(loc=gamma_big, scale=lam_sd)
    lam_dens = -jnp.sum(lam_dist.log_prob(x)-jnp.log(1-lam_dist.cdf(0)))
    return lam_dens + gamma_dens

@jit
def get_nll(params, lam, log_gamma, X, y):
    pred = predict(params, X.T)
    nll = -jnp.sum(tfpd.Categorical(logits=pred.T).log_prob(y))
    nll += nn_prior(lam, log_gamma, X)
    nll += -jnp.sum(jnp.log(lam))## LAM OFF
    return nll

grad = jax.grad(get_nll, argnums = [0,1,2])

def update_raw(params, lam, log_gamma, lam_step, tau0, X, y):
  grads, lam_grad, gam_grad = grad(params, lam, log_gamma, X, y)
  new_params = [(w - step_size * dw, b - step_size * db) for (w, b), (dw, db) in zip(params, grads)]
  new_lam = lam - lam_step * lam_grad
  new_gam = log_gamma - lam_step * gam_grad
  #if not TAU_ZERO:
  #    print("Doing prox")
  new_W, new_lam = jax_apply_prox(new_params[0][0], new_lam, tau0*step_size, tau0*lam_step) ## LAM OFF
  new_params[0] = (new_W, new_params[0][1])
  #else:
  #    print("no prox")
  return(new_params, new_lam, new_gam)


def train_net(X, y, tau0):
    X = (X-np.mean(X,axis=0)[np.newaxis,:])/np.std(X,axis=0)[np.newaxis,:]
    y = np.array(y)

    update = jax.jit(update_raw)

    layer_sizes = [X.shape[1], 2, 512, 3]

    params = init_network_params(layer_sizes, jax.random.PRNGKey(seed))
    lam = lam_init * jnp.ones_like(params[0][0])
    log_gamma = jnp.zeros_like(params[0][0][0,:])

    lam_step = lam_init_step

    costs = np.zeros(num_epochs)
    for i in tqdm(range(num_epochs)):
        if i == lam1at:
          lam = jnp.ones_like(lam)
          lam_step = lam_eventual_step
        cost = get_nll(params, lam, log_gamma, X, y)
        costs[i] = cost
        params, lam, log_gamma = update(params, lam, log_gamma, lam_step, tau0, X, y)

    return params, costs
