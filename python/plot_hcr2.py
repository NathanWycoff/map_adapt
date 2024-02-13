#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  plot_hcr2.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 02.09.2024

import pickle
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import medfilt

exec(open('python/hcr_settings.py').read())

with open("pickles/traj_hcr_"+str(eu_only)+'.pdf', 'rb') as f:
    df_means, df_zeros, resdf = pickle.load(f)

for i in range(len(df_means)):
    df_means[i].columns = [np.flip(tau_range)[i], 'name']

res = df_means[0]
for i in range(1,len(df_means)):
    res = pd.merge(res,df_means[i], how = 'outer', on = 'name')

res.index = res.name
res = res.drop('name',axis=1)
res = res.fillna(0)

res = res.iloc[:,:90]
#res = res.iloc[:,:85]

res = np.maximum(-0.5, np.minimum(0.5, res))

fig = plt.figure(figsize=[5,2.5])
K = 3
for i in res.index:
    #filt = np.zeros(res.shape[1])
    #for t in range(res.shape[1]):
    #    filt[t] = np.max(res.loc[i,:].iloc[t:(t+K)])
    #plt.plot(res.columns, filt)
    plt.plot(res.columns, medfilt(res.loc[i,:],7))
plt.xscale('log')
plt.savefig('traj.pdf')
plt.close()

