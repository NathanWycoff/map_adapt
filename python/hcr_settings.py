import numpy as np

hcr_iters = 1 # really reps not iters.
adam = True
prop_train = 0.5
decay_learn = True

#eu_only = True
eu_only = False

## Full run.
if eu_only:
    max_iters = 5000
    n_tau = 100
else:
    max_iters = 3200
    #n_tau = 100
    #n_tau = 25
    n_tau = 50

## Short run.
##max_iters = 2500
#max_iters = 500
#n_tau = 10

es_patience = np.inf

#max_iters = 10
mb_size = 256
ada = True

#big_boi = True #Use quadratic model? 
expansion = 'intr' 
big_boi = expansion in ['intr','quad']
synthetic = False

if eu_only:
    #tau_range = np.logspace(np.log10(750),5,num=n_tau)
    tau_range = np.logspace(np.log10(750),4.2,num=n_tau)
    lr = 2e-3
    #lr = 1e-2
else:
    #tau_range = np.logspace(np.log10(750),5,num=n_tau)
    tau_range = np.logspace(np.log10(75000),5,num=n_tau)
    #tau_range = np.logspace(5,6,num=n_tau)
    lr = 1e-3

simout_dir = 'hcr_eu/' if eu_only else 'hcr_global' 
