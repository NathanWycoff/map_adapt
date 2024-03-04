#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  table_size.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 02.23.2024

## Structured selection with hierarchical models a la Roth and Fischer
import pickle
import numpy as np
import scipy
from matplotlib.gridspec import GridSpec
import pandas as pd
from time import time
import statsmodels.api as sm
import sys
from scipy.special import comb

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

exec(open('python/sim_settings.py').read())

table = pd.DataFrame(np.zeros([len(datasets_to_use), 5]).astype(int))
table.columns = ['Problem','N','P','P2nd','Likelihood']
table['Problem'] = [x.title() for x in datasets_to_use]
table.index = datasets_to_use

for s_i in datasets_to_use:
    df = pd.read_csv(data_dir+s_i+'.csv')
    X_all = np.array(df.iloc[:,:-1])

    table.loc[s_i,'N'] = X_all.shape[0]
    P = X_all.shape[1]
    table.loc[s_i,'P'] = P
    table.loc[s_i,'P2nd'] = 2*P + comb(P,2)
    table.loc[s_i,'Likelihood'] = liks[s_i].title()

table = table.sort_values('N')

with open('tables/prob_size.txt','w') as f:
    table.to_latex(buf=f, index = False)

