## Structured selection with hierarchical models a la Roth and Fischer
import pickle
import numpy as np
import scipy
from matplotlib.gridspec import GridSpec
import pandas as pd
from time import time
import statsmodels.api as sm
import sys
import matplotlib.pyplot as plt
from jax.scipy.special import logsumexp
from tqdm import tqdm
import matplotlib.patches as mpatches

#seed = int(time()*100) % 10000
seed = 1900
np.random.seed(seed)

exec(open('python/jax_nsa.py').read())
exec(open('python/jax_hier_lib.py').read())
exec(open('python/sim_lib.py').read())
exec(open('python/sim_settings.py').read())
exec(open('python/glmnet_wrapper.py').read())

analysis = 'lowpenalty'
#analysis = 'highpenalty'

if analysis == 'lowpenalty':
    step_size = 1e-3
    tau0 = 1e1
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

#an = 'flips'
an = 'outcomes'

if an=='flips':
    dat = pd.read_csv("data/flips_dat.csv")
    Xl = dat.iloc[:,1:-3]
    X = np.array(Xl)
    y = np.array(dat['flipped'].astype(float))
elif an=='outcomes':
    dat = pd.read_csv("data/outcomes_dat.csv")
    Xl = dat.iloc[:,4:-1]
    X = np.array(Xl)
    y = dat[['EA','VS','PS']].apply(lambda x: np.where(x)[0][0], axis = 1)

X = (X-np.mean(X,axis=0)[np.newaxis,:])/np.std(X,axis=0)[np.newaxis,:]

# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = jax.random.split(key)
  return scale * jax.random.normal(w_key, (n, m)), scale * jax.random.normal(b_key, (n,))

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
  keys = jax.random.split(key, len(sizes))
  return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

layer_sizes = [X.shape[1], 2, 512, 3]

from time import time
params = init_network_params(layer_sizes, jax.random.PRNGKey(seed))
lam = lam_init * jnp.ones_like(params[0][0])
log_gamma = jnp.zeros_like(params[0][0][0,:])

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

def nn_prior(x, log_gamma):
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
def get_nll(params, lam, log_gamma):
    pred = predict(params, X.T)
    nll = -jnp.sum(tfpd.Categorical(logits=pred.T).log_prob(y))
    nll += nn_prior(lam, log_gamma)
    nll += -jnp.sum(jnp.log(lam))## LAM OFF
    return nll

grad = jax.grad(get_nll, argnums = [0,1,2])

@jit
def update(params, lam, log_gamma, lam_step):
  grads, lam_grad, gam_grad = grad(params, lam, log_gamma)
  new_params = [(w - step_size * dw, b - step_size * db) for (w, b), (dw, db) in zip(params, grads)]
  new_lam = lam - lam_step * lam_grad
  new_gam = log_gamma - lam_step * gam_grad
  if tau0 > 0:
      new_W, new_lam = jax_apply_prox(new_params[0][0], new_lam, tau0*step_size, tau0*lam_step) ## LAM OFF
      new_params[0] = (new_W, new_params[0][1])
  return(new_params, new_lam, new_gam)

lam_step = lam_init_step

costs = np.zeros(num_epochs)
for i in tqdm(range(num_epochs)):
    if i == lam1at:
      lam = jnp.ones_like(lam)
      lam_step = lam_eventual_step
    cost = get_nll(params, lam, log_gamma)
    costs[i] = cost
    params, lam, log_gamma = update(params, lam, log_gamma, lam_step)

fig = plt.figure()
plt.plot(costs)
plt.savefig("costs.pdf")
plt.close()

Z = (params[0][0] @ X.T).T

ng = 75
rx = [np.min(Z[:,0]), np.max(Z[:,0])]
gx = np.linspace(rx[0], rx[1], num = ng)
ry = [np.min(Z[:,1]), np.max(Z[:,1])]
gy = np.linspace(ry[0], ry[1], num = ng)

G = np.zeros([ng*ng,2])
for i in range(ng):
   for j in range(ng):
      G[i*ng+j,0] = gx[i]
      G[i*ng+j,1] = gy[j]

def predict_latent(params, Z):
  # per-example predictions
  activations = Z
  for w, b in params[1:-1]:
    #outputs = jnp.dot(w, activations) + b
    outputs = w @ activations + b[:,jnp.newaxis]
    activations = relu(outputs)
  
  final_w, final_b = params[-1]
  logits = final_w @ activations + final_b[:,jnp.newaxis]
  return logits

out = np.array(predict_latent(params, G.T).T)
gclass = np.apply_along_axis(lambda x: np.argmax(x), 1, out)
gcols = [['green','blue','red'][yi] for yi in gclass]


##### Left Column of Deep Active Classifier Figure
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
fig = plt.figure(figsize=[4,4])

plt.scatter(G[:,0],G[:,1], color = gcols, s = 10, marker = 's', alpha = 0.1)
cols = [['green','blue','red'][yi] for yi in y]
jitter = np.random.normal(size=Z.shape, scale = 2e-2)
alpha = [0.6 if c =='green' else 1. for c in cols]
plt.title(r"Deep Active Subspace; $\tau=$"+str(tau0))

if analysis=='lowpenalty':
  patches = []
  patches.append(mpatches.Patch(color='green', label='Early Adopter'))
  patches.append(mpatches.Patch(color='blue', label='Vaccinated Skeptic'))
  patches.append(mpatches.Patch(color='red', label='Persistent Antivaxer'))
  #plt.legend(handles=patches, prop = {'size':6}, framealpha = 1.)
  plt.legend(handles=patches, prop = {'size':8}, framealpha = 1., loc = 'upper left')

plt.scatter(Z[:,0]+jitter[:,0],Z[:,1]+jitter[:,1], color = cols, s = 5, alpha = alpha)
plt.tight_layout()
plt.savefig("proj_"+analysis+str(np.round(tau0,1))+".pdf")
plt.close()
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

print(params[0][0])
print(lam)
print(log_gamma)

BETA = params[0][0]

K = 100 # Average over this many individuals when making plots.


##### Right Column of Deep Active Classifier Figure
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
fig = plt.figure(figsize=[5,4])
lsize = 5

politicians = ['tedcruz','BarackObama','DonaldJTrumpJr','POTUS']
entertainment = ['KendallJenner','Drake','SportsCenter','TheRealMikeEpps','NFL','KevinHart4real']
demographics = ['parent', 'race_Black', 'edred_<4y', 'POWNHOME_Rented']
plot_vars = politicians + entertainment + demographics 

import matplotlib as mpl
cm = mpl.colormaps['tab10']

var_traj = {}
for ci, coord in enumerate(['x','y']):
    gg = gx if coord=='x' else gy

    var_traj[coord] = pd.DataFrame(np.zeros([gg.shape[0], len(plot_vars)]))
    var_traj[coord].index = gg
    var_traj[coord].columns = plot_vars

    for i in range(gg.shape[0]):
        dists = np.square(Z[:,ci]-gg[i])
        idx = np.argpartition(dists, K)[:K]
        for v in plot_vars:
            var_traj[coord].loc[gg[i],v] = np.mean(Xl.loc[idx,v])

for v in plot_vars:
   mu_x = (np.mean(var_traj['x'][v]) + np.mean(var_traj['y'][v])) / 2
   sig_x = np.sqrt((np.var(var_traj['x'][v]) + np.var(var_traj['y'][v])) / 2)
   var_traj['x'].loc[:,v] = (var_traj['x'].loc[:,v] - mu_x) / sig_x
   var_traj['y'].loc[:,v] = (var_traj['y'].loc[:,v] - mu_x) / sig_x

for ci, coord in enumerate(['x','y']):
    ii = 1 if ci==0 else 2
    plt.subplot(3,2,ii)
    plt.title("Political Figures " + coord.upper() + '-Axis')
    for vi,v in enumerate(politicians):
        plt.plot(var_traj[coord].loc[:,v], label = v, color = cm(vi/len(politicians)))

    ii = 3 if ci==0 else 4
    plt.subplot(3,2,ii)
    plt.title("Entertainment " + coord.upper() + '-Axis')
    for vi,v in enumerate(entertainment):
        plt.plot(var_traj[coord].loc[:,v], label = v, color = cm(vi/len(entertainment)))

    ii = 5 if ci==0 else 6
    plt.subplot(3,2,ii)
    plt.title("Demographics " + coord.upper() + '-Axis')
    for vi,v in enumerate(demographics):
        plt.plot(var_traj[coord].loc[:,v], label = v, color = cm(vi/len(demographics)))

plt.tight_layout()
plt.savefig('inv_reg_'+analysis+str(np.round(tau0,1))+'.pdf')
plt.close()


fig = plt.figure(figsize=[1.8,4])

for ti,target in enumerate([politicians, entertainment, demographics]):
  plt.subplot(3,1,ti+1)
  patches = []
  for vi,v in enumerate(target):
    patches.append(mpatches.Patch(color=cm(vi/len(target)), label=v))
  plt.legend(loc=['upper left','center left','lower left'][ti],handles=patches, prop = {'size':8})
  #plt.legend('bottom left',handles=patches)

  plt.axis("off")

plt.tight_layout()
plt.savefig('mosaic_nn_legend.pdf')
plt.close()
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
# Benchmarks

svd = np.linalg.svd(X)
Z = svd[0][:,:2] @ np.diag(svd[1][:2])
#(X @ svd[2].T[:,:2])

cols = [['green','blue','red'][yi] for yi in y]
alpha = [0.6 if c =='green' else 1. for c in cols]

ng = 75
rx = [np.min(Z[:,0]), np.max(Z[:,0])]
gx = np.linspace(rx[0], rx[1], num = ng)
ry = [np.min(Z[:,1]), np.max(Z[:,1])]
gy = np.linspace(ry[0], ry[1], num = ng)

G = np.zeros([ng*ng,2])
for i in range(ng):
   for j in range(ng):
      G[i*ng+j,0] = gx[i]
      G[i*ng+j,1] = gy[j]

#from sklearn.neighbors import NearestNeighbors
from sklearn import neighbors
n_neighbors = 5
clf = neighbors.KNeighborsClassifier(n_neighbors)

fit = clf.fit(Z, y)
gclass = fit.predict(G)
#gclass = np.apply_along_axis(lambda x: np.argmax(x), 1, out)
gcols = [['green','blue','red'][yi] for yi in gclass]


##### Left Column of Deep Active Classifier Figure
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
fig = plt.figure(figsize=[4,4])
plt.scatter(Z[:,0],Z[:,1], color=cols, s = 5, alpha = alpha)
plt.scatter(G[:,0],G[:,1], color = gcols, s = 10, marker = 's', alpha = 0.1)
plt.title("PCA")
plt.savefig("mosaic_pca.pdf")
plt.close()


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
l = svd[2].T@np.diag(1 /svd[1])
Xc = X @ l
np.linalg.svd(Xc)

x0 =np.mean(Xc[y == 0, :], axis=0)
x1 =np.mean(Xc[y == 1, :], axis=0)
x2 =np.mean(Xc[y == 2, :], axis=0)

n0 = np.sum(y == 0)
n1 = np.sum(y == 1)
n2 = np.sum(y == 2)

x_sir = np.stack([n0 *x0, n1 *x1, n2 *x2]) 
np.mean(x_sir)
svd = np.linalg.svd(x_sir)
eta =svd[2][:2, :].T
beta = l @ eta
Z = X @ beta


ng = 75
rx = [np.min(Z[:,0]), np.max(Z[:,0])]
gx = np.linspace(rx[0], rx[1], num = ng)
ry = [np.min(Z[:,1]), np.max(Z[:,1])]
gy = np.linspace(ry[0], ry[1], num = ng)

G = np.zeros([ng*ng,2])
for i in range(ng):
   for j in range(ng):
      G[i*ng+j,0] = gx[i]
      G[i*ng+j,1] = gy[j]

#from sklearn.neighbors import NearestNeighbors
from sklearn import neighbors
n_neighbors = 5
clf = neighbors.KNeighborsClassifier(n_neighbors)

fit = clf.fit(Z, y)
gclass = fit.predict(G)
#gclass = np.apply_along_axis(lambda x: np.argmax(x), 1, out)
gcols = [['green','blue','red'][yi] for yi in gclass]


cols = [['green','blue','red'][yi] for yi in y]
alpha = [0.6 if c =='green' else 1. for c in cols]

fig = plt.figure(figsize=[4,4])
plt.scatter(Z[:,0],Z[:,1], color=cols, s = 5, alpha = alpha)
plt.scatter(G[:,0],G[:,1], color = gcols, s = 10, marker = 's', alpha = 0.1)
plt.title("SIR")
plt.savefig("mosaic_sir.pdf")
plt.close()
