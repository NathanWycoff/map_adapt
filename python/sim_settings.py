#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  sim_settings.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.19.2023

## Params
NN = 1000 #TODO: overwritten if libsvm.

iters = 30
#iters = 2
Ns = [10000]

#Pu = 200
#Pnz = 20 

# setting 5-9 did not finish.

sim = 'synthetic'
#sim = 'libsvm'
#beta_style = 'random'
beta_style = 'floor'
for i in range(10):
    print(beta_style)
#beta_style = 'fixed'

max_iters = 10000
mb_size = 256
es_patience = 500

#sparsity_type = 'random' # Totally random sparsity
sparsity_type = 'group' #Group sparsity 
#sparsity_type = 'hier2nd' #Overlapping group sparsity for hierarchical model.
if sparsity_type=='random':
    lr = 1e-2
    #lr = 1e-1
elif sparsity_type=='group':
    #lr = 1e-3
    lr = 1e-2
else:
    raise NotImplementedError()

usejags = False
JAGS_MAX_N = 100 # If n>this, we only do JAGS_N_REPS versions of JAGS.
if sparsity_type=='random':
    JAGS_N_REPS = 5
else:
    JAGS_N_REPS = 0

if sparsity_type=='random':
    models2try = ['sbl_ada','glmnet','OLS']
elif sparsity_type=='group':
    models2try = ['sbl_ada','sbl_group','glmnet','MLGL','OLS']
elif sparsity_type=='hier2nd':
    models2try = ['sbl_hier','glmnet','MLGL','OLS']
else:
    raise Exception

nice_names = {
    'ida_net':'ida et al 2017',
    'sbl_ada' : 'Lap-Mix-Ind',
    'sbl_group' : 'Lap-Mix-Group',
    'sbl_hier' : 'Lap-Mix-Hier',
    'glmnet' : 'glmnet',
    'MLGL' : 'Grimonprez et al 2021',
    'OLS' : 'GLM',
    'jags' : 'JAGS-Horseshoe'
}

model_colors = {'OLS' : 'green',
                'glmnet' : 'cyan',
                'MLGL': 'blue',
                'jags': 'gray',
                'ida_net' : 'purple',
                'sbl_ada' : 'red',
                'sbl_group' : 'orange',
                'sbl_hier' : 'pink'}

if usejags:
    models2try += ['jags']

settings = []
for N in Ns:
    if N == 10000:
        if sparsity_type == 'random':
            Pu = 1000
        elif sparsity_type == 'group':
            Pu = 1000//5
        else:
            raise NotImplementedError()
        #print("5k!")
        #Pu = 5000
        Pnz = 5
        #Pnz = 1
        #Pnz = 20
        for i in range(10):
            print("PNZ %d!"%Pnz)
    elif N == 10000:
        Pu = 20
        Pnz = 2
    else:
        raise Exception("N bad.")
    #settings += [(N,Pu,Pnz,'normal', 1e0),(N,Pu,Pnz,'normal', 1e-1),(N,Pu,Pnz,'normal', 1e-2),]+\
    #        [(N,Pu,Pnz,'poisson', 1e0),(N,Pu,Pnz,'cauchy', 1e0),(N,Pu,Pnz,'bernoulli', 1e0)]
    settings += [(N,Pu,Pnz,'normal', 1e0),(N,Pu,Pnz,'poisson', 1e0),(N,Pu,Pnz,'cauchy', 1e0),(N,Pu,Pnz,'bernoulli', 1e0)]

reg_problems = ['abalone','housing','bodyfat','mpg','triazines','mg','hcr_all','hcr_eu']
class_problems = ['diabetes','australian','heart','covtype']
datasets_to_use = reg_problems + class_problems
liks = {
    'abalone' : 'poisson',
}
for v in class_problems:
    liks[v] = 'bernoulli'
for v in set(datasets_to_use).difference(liks):
    liks[v] = 'normal'

if sim=='synthetic':
    qoi = ['beta-MSE','yy-MSE','FPR','FNR']
elif sim == 'libsvm':
    qoi = ['yy-MSE','nonzero']
else:
    raise Exception("sim unrecognized.")

simout_dir = sim+"_"+sparsity_type
