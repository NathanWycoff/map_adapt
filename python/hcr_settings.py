import numpy as np

#iters = 200
#iters = 10000
hcr_iters = 3
#lr = 0.1
#lr = 1e-2
adam = True
#adam = False
prop_train = 0.5
decay_learn = True

#reps = 10
#n_tau = 5
#n_tau = 2
n_tau = 50

max_iters = 20000
#max_iters = 10
mb_size = 256
es_patience = 500
ada = True

big_boi = True #Use quadratic model? 
synthetic = False
eu_only = True

simout_dir = 'hcr_eu/' if eu_only else 'hcr_global' 
