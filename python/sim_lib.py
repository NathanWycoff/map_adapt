#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  sim_lib.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.19.2023

import numpy as np

def random_sparsity(P,Pnz):
    nonzero = np.zeros(P)
    nonzero[np.random.choice(P,Pnz)] = 1.

    return nonzero

def group_sparsity(Pu,Pnz):
    nonzero = np.zeros(P)
    nonzero[np.random.choice(P,Pnz)] = 1.

    return nonzero

# The matrix Gamma links main effects to the interaction terms including them.
# Each row, of which there are P, gives indices of the interaction terms which it belongs to.
def make_gamma_mat(P):
    tops = []
    bots = []
    for i in range(P):
        if i > 0:
            incs = np.concatenate([[0],np.cumsum(P-1-np.arange(1,i))])
            tops.append(i+incs)
        else:
            tops.append([])
    for i in range(P-1):
        if i >0:
            startat = tops[i+1][i]
        else:
            startat = 1
        #print(startat)
        bots.append(np.arange(startat,startat+P-1-i))
    bots.append([])
    gammat = np.zeros([P-1,P])
    for i in range(P):
        gammat[:,i] = np.concatenate([tops[i], bots[i]])
    gammat -= 1
    return gammat

def add_int_quad(Xu, var_names = None):
    N,Pu = Xu.shape

    if var_names is None:
        var_names = ['X'+str(i) for i in range(Pu)]

    Pi = int(scipy.special.binom(Pu,2))
    Pq = Pu
    P = Pu + Pi + Pq

    # Interactions
    Xi = np.zeros([N,Pi])
    ind = 0
    int_name = []
    for i in range(Pu-1):
        for j in range(i+1,Pu):
            Xi[:,ind] = Xu[:,i]*Xu[:,j]
            ind += 1
            int_name.append(var_names[i]+'-'+var_names[j])

    # Quadratics
    Xq = np.zeros([N,Pq])
    ind = 0
    quad_name = []
    for i in range(Pu):
        Xq[:,i] = np.square(Xu[:,i])
        quad_name.append(var_names[i]+'^2')

    X = np.concatenate([Xu,Xi,Xq], axis = 1)
    Xdf = pd.DataFrame(X)
    Xdf.columns = var_names + int_name + quad_name

    return Xdf


def hier2nd_sparsity(Pu,Pnz):
    ngroups = int(scipy.special.binom(Pu,2))
    P = 2*Pu + ngroups

    group_sparsity = np.zeros(ngroups)
    group_sparsity[np.random.choice(ngroups,Pnz)] = 1.
    #group_sparsity = np.random.binomial(1,p_groupinc,size=ngroups)
    int_sparsity = group_sparsity

    #TODO: Verify
    GG = make_gamma_mat(Pu).astype(int)
    GAMMA = group_sparsity[GG]
    me_sparsity = np.any(GAMMA, axis = 0).astype(float)
    q_sparsity = me_sparsity

    #v1 = np.repeat(np.arange(Pu-1), np.flip(np.arange(1,Pu)))
    #v2 = np.repeat(np.arange(1,Pu), np.arange(1,Pu))
    v1 = np.zeros(ngroups).astype(int)
    v2 = np.zeros(ngroups).astype(int)
    ii = 0
    for i in range(Pu-1):
        for j in range(i+1,Pu):
            v1[ii] = i
            v2[ii] = j
            ii += 1

    nonzero = np.concatenate([me_sparsity, int_sparsity, q_sparsity])

    #flipendo = np.zeros(P)
    #flipendo[np.random.choice(P,2)] = 1.

    #nonzero = (1-flipendo) * nonzero + flipendo * (1-nonzero)

    return nonzero, ngroups, P, v1, v2