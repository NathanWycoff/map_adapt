#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  sim_settings.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.19.2023

import numpy as np

## Params
NN = 1000 #TODO: overwritten if libsvm.

iters = 30
#iters = 4
#iters = 2
Ns = [10000]

#Pu = 200
#Pnz = 20 

# setting 5-9 did not finish.

sim = 'synthetic'
#sim = 'libsvm'
beta_style = 'random'
#beta_style = 'floor'
#beta_style = 'fixed'

max_iters = 20000
mb_size = 256
es_patience = 500
ada = True
#ada = False

#sparsity_type = 'random' # Totally random sparsity
sparsity_type = 'group' #Group sparsity 
#sparsity_type = 'hier2nd' #Overlapping group sparsity for hierarchical model.
if sparsity_type=='random':
    lr = 1e-2
elif sparsity_type=='group':
    lr = 1e-3
elif sparsity_type=='hier2nd':
    lr = 1e-3
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
    models2try = ['sbl_ada','sbl_hier','glmnet','MLGL','OLS']
    #models2try = ['sbl_ada','sbl_hier','glmnet','OLS']
    #models2try = ['sbl_ada','sbl_hier','OLS']
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
        elif sparsity_type == 'hier2nd':
            Pu = int(np.sqrt(2*1000))
        else:
            raise NotImplementedError()
        Pnz = 10
    elif N == 10000:
        Pu = 20
        Pnz = 2
    else:
        raise Exception("N bad.")
    #settings += [(N,Pu,Pnz,'normal', 1e0),(N,Pu,Pnz,'normal', 1e-1),(N,Pu,Pnz,'normal', 1e-2),]+\
    #        [(N,Pu,Pnz,'poisson', 1e0),(N,Pu,Pnz,'cauchy', 1e0),(N,Pu,Pnz,'bernoulli', 1e0)]
    #settings += [(N,Pu,Pnz,'normal', 1e0),(N,Pu,Pnz,'poisson', 1e0),(N,Pu,Pnz,'cauchy', 1e0),(N,Pu,Pnz,'bernoulli', 1e0)]
    settings += [(N,Pu,Pnz,'normal', 1e0),(N,Pu,Pnz,'nb', 1e0),(N,Pu,Pnz,'cauchy', 1e0),(N,Pu,Pnz,'bernoulli', 1e0)]

reg_problems = ['abalone','housing','bodyfat','mpg','triazines','mg','hcr_all','hcr_eu','kin40k','keggu','year','bike','obesity','seoul','parkinsons','infra','wine']
class_problems = ['diabetes','australian','heart','covtype','mushrooms','phishing','aids','rice','adult','spam','dropout','shop']
#datasets_to_use = reg_problems + class_problems
#datasets_to_use = ['abalone','housing','bodyfat','mpg','mg','diabetes','heart','covtype','mushrooms','phishing']
#datasets_to_use = ['housing','bodyfat','mpg','mg','diabetes','australian','heart','covtype','mushrooms','phishing']
#datasets_to_use = ['housing','bodyfat','mpg','mg','diabetes','australian','heart','covtype','phishing']
#datasets_to_use = ['kin40k','bodyfat','abalone','keggu','diabetes','australian','year','covtype','phishing']
#datasets_to_use = ['kin40k','year','keggu']
#datasets_to_use = reg_problems + class_problems
#datasets_to_use = ['abalone','bike','obesity','seoul','parkinsons','aids','infra','rice','adult','wine','spam','dropout','shop']

datasets_to_use = ['abalone','obesity','parkinsons','aids','rice','spam','dropout','shop']

#datasets_to_use = ['abalone','bike','diabetes','']
#liks = {
#        #'abalone' : 'poisson',
#        'abalone' : 'nb',
#}
liks = {}
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

libsvm_dir = "./libsvm_data/"
data_dir = "./data/"
simout_dir = sim+"_"+sparsity_type
