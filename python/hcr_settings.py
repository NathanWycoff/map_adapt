import numpy as np

#iters = 200
#iters = 10000
#hcr_iters = 3
hcr_iters = 1 # really reps not iters.
#lr = 0.1
#lr = 1e-2
adam = True
#adam = False
prop_train = 0.5
decay_learn = True

#reps = 10
#n_tau = 5
#n_tau = 2
#n_tau = 50

#tau_range = np.logspace(2,np.log10(774.263683),num=n_tau)
max_iters = 2500
#max_iters = 500
n_tau = 100
#n_tau = 10
#tau_range = np.logspace(1,4,num=n_tau)
#tau_range = np.logspace(1,np.log10(50),num=n_tau)
#tau_range = np.logspace(1,np.log10(2*50),num=n_tau)
#tau_range = np.logspace(2,3,num=n_tau)
#tau_range = np.logspace(1,2*np.log10(50),num=2*n_tau)
#tau_range = np.logspace(3,5,num=n_tau)
tau_range = np.logspace(2.5,5,num=n_tau)
es_patience = np.inf

#max_iters = 10
mb_size = 256
ada = True

big_boi = True #Use quadratic model? 
synthetic = False
eu_only = True

simout_dir = 'hcr_eu/' if eu_only else 'hcr_global' 
