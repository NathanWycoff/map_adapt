#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  hcr_lib.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 02.08.2024

# Structured selection with hierarchical models a la Roth and Fischer
import pickle
import numpy as np
import scipy
from matplotlib.gridspec import GridSpec
import pandas as pd
from time import time
import statsmodels.api as sm
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_data(expansion, synthetic, eu_only, lik='zinb', dump_top = False, random_effects = True, prop_train = 0.5, norm = True):
    assert expansion in ['none','intr','quad']
    df = pd.read_csv('./data/hcr_impu1.csv').iloc[:, 1:]

    big_boi = expansion in ['intr','quad']
    synthetic_interact = big_boi

    ## Keep only european countries 
    europe_iso = [
    'ALB','AND','AUT','BLR','BEL','BIH','BGR','HRV','CYP','CZE','DNK','EST','FIN','FRA','DEU','GRC','HUN','ISL','IRL','ITA','KOS','LVA','LIE','LTU','LUX','MKD','MLT','MDA','MCO','MNE','NLD','NOR','POL','PRT','ROU','RUS','SMR','SRB','SVK','SVN','ESP','SWE','CHE','TUR','UKR','GBR','VAT'
    ]
    #europe_iso = europe_iso[:(len(europe_iso)//5)]
    #europe_iso = europe_iso[:(len(europe_iso)//2)]

    df['dist'] = np.log(df['dist'])
    eu2eu = np.logical_and(df.iso_d.isin(europe_iso), df.iso_o.isin(europe_iso))
    if eu_only:
        df = df.loc[eu2eu,:]

    ## Drop redundant death vars.
    df = df.drop(['dead_log_o','dead_o','dead_log_d','dead_d'], axis = 1)
    
    ## Log some vars
    letslog = ['area','pop','best_est','Nyear_conflict']
    for dy in ['_o','_d']:
        v = [x+dy for x in letslog]
        df[v] = np.log10(df[v]+1.)

    realx = 'dist'
    if synthetic:
        if synthetic_interact:
            lmu = df['pop_o']/np.max(df['pop_o']) + df['dist'] / np.max(df['dist']) - df['pop_o']/np.max(df['pop_o'])*df['dist'] / np.max(df['dist'])
            lmu *= 10
            if not big_boi:
                print("Warning: using interaction data in main effects model.")
        else:
            lmu = df['pop_o']/np.max(df['pop_o']) + df['dist'] / np.max(df['dist'])
            lmu *= 5
        if lik in ['poisson','nb','zinb']:
            y_dist = tfpd.Poisson(log_rate=lmu)
        elif lik=='normal':
            y_dist = tfpd.Normal(loc=lmu, scale=1.)
        else:
            raise NotImplementedError()
        y = y_dist.sample(seed=key)
        print(np.max(y))
    else:
        y = np.array(df['newarrival'])
        if lik == 'normal':
            y = np.log(y+1)

    marg_vars = [x for x in df.columns[3:] if (
        x[-2:] in ['_o', '_d'] and x not in ['Country_o', 'Country_d'])]
    dy_vars = list(df.columns[-8:])
    xcols = np.array(marg_vars+dy_vars)
    Xd = df[xcols]

    X = np.array(Xd)
    Xi = X.copy()
    if norm:
        X = (X - np.mean(X, axis=0)[np.newaxis, :]) / np.std(X, axis=0)[np.newaxis, :]

    n_train = int(np.ceil(prop_train * X.shape[0]))

    if random_effects:
        B = df.loc[:, ['iso_o', 'iso_d', 'year']]
        B['year'] = B['year'].astype(str)
        # Bd = pd.get_dummies(B, drop_first=True)
        Bd = pd.get_dummies(B, drop_first=False)
        X = np.concatenate([X, Bd], axis=1)
        # dont_pen = np.arange(Xi.shape[1], X.shape[1])
        # if lik == 'zinb':
        #    dont_pen = np.concatenate([dont_pen, dont_pen+X.shape[1]])
    dont_pen = np.array([]).astype(int)

    #re_names = [x+'_o' for x in list(set(B['iso_o']))] + \
    #    [x+'_d' for x in list(set(B['iso_d']))]+list(set(B['year']))
    re_names = list(Bd.columns)
    av_names = np.concatenate([xcols, re_names])

    Xempty = X[:0, :]
    if expansion=='intr':
        Xempty_big = add_int(Xempty, var_names=list(av_names))
        print("A")
    else:
        Xempty_big = add_int_quad(Xempty, var_names=list(av_names))
        print("B")
    n_re = len(set(df['iso_d'])) + len(set(df['iso_d'])) + len(set(df['year']))
    #Xempty_big = Xempty_big.iloc[:, :-n_re]
    av_names_big = Xempty_big.columns
    # Xd
    yempty = np.array([])

    me_names = av_names_big[:X.shape[1]]
    int_names = av_names_big[X.shape[1]:-X.shape[1]]
    qu_names = av_names_big[-Xd.shape[1]:]

    if dump_top:
        top_K = 200
        top_y = np.argpartition(y, -top_K)[-top_K:]
        keep_y = np.setdiff1d(np.arange(len(y)),top_y)
        y = y[keep_y]
        X = X[keep_y,:]

    # for rep in range(reps):
    inds_train = np.random.choice(X.shape[0], n_train, replace = False)
    inds_test = np.delete(np.arange(X.shape[0]), inds_train)
    X_train = X[inds_train, :]
    y_train = y[inds_train]
    X_test = X[inds_test, :]
    y_test = y[inds_test]

    return X_train, y_train, X_test, y_test, xcols, re_names, av_names_big
