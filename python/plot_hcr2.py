#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  plot_hcr2.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 02.09.2024

import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from scipy.signal import medfilt
from adjustText import adjust_text
import glob

exec(open('python/hcr_settings.py').read())

with open("pickles/traj_hcr_"+str(eu_only)+'.pkl', 'rb') as f:
    df_means, df_zeros, resdf = pickle.load(f)

for i in range(len(df_means)):
    df_means[i].columns = [np.flip(tau_range)[i], 'name']
for i in range(len(df_zeros)):
    df_zeros[i].columns = [np.flip(tau_range)[i], 'name']

res = df_means[0]
res0 = df_zeros[0]
for i in range(1,len(df_means)):
    res = pd.merge(res,df_means[i], how = 'outer', on = 'name')
    res0 = pd.merge(res0,df_zeros[i], how = 'outer', on = 'name')

res.index = res.name
res = res.drop('name',axis=1)
res = res.fillna(0)
res0.index = res0.name
res0 = res0.drop('name',axis=1)
res0 = res0.fillna(0)

medrad = 7
#medrad = 3
for ii,v in enumerate(res.index):
    res.loc[v,:] =  medfilt(res.loc[v,:],medrad)
for ii,v in enumerate(res0.index):
    res0.loc[v,:] =  medfilt(res0.loc[v,:],medrad)

# To aid visualization keep everthing close.
thresh = 0.75
res = np.maximum(-thresh, np.minimum(thresh, res))
res0 = np.maximum(-thresh, np.minimum(thresh, res0))

##### Get first nonzero and order accordingly
first_nz = np.zeros(res.shape[0]).astype(int)
for i,v in enumerate(res.index):
    if np.any(res.loc[v,:]!=0):
        first_nz[i] = np.where(res.loc[v,:] != 0)[0][0]
    else:
        first_nz[i] = res.shape[1]-1
label_K = 5
first_inds = np.argpartition(-first_nz, -label_K)[-label_K:]
first_nz[first_inds]
order = [x for _, x in sorted(zip(first_nz, np.arange(res.shape[0])))]
res = res.iloc[order,:]

first_nz0 = np.zeros(res0.shape[0]).astype(int)
for i,v in enumerate(res0.index):
    if np.any(res0.loc[v,:]!=0):
        first_nz0[i] = np.where(res0.loc[v,:] != 0)[0][0]
    else:
        first_nz0[i] = res0.shape[1]-1
label_K = 5
first_inds = np.argpartition(-first_nz0, -label_K)[-label_K:]
first_nz0[first_inds]
order = [x for _, x in sorted(zip(first_nz0, np.arange(res0.shape[0])))]
res0 = res0.iloc[order,:]
##### Get first nonzero and order accordingly


xlim = 40
res = res.iloc[:,:xlim]
res0 = res0.iloc[:,:xlim]

nll_ind = 35
print("nll:")
print(resdf['nll'][nll_ind])

#xlim = 80
#xlim = 100
#xlim = 50

#fig = plt.figure(figsize=[5,2.5])
fig = plt.figure(figsize=[5,5])

plt.subplot(2,1,1)
cnt = 0
#nbigm = max(x.shape[0] for x in df_means)
nbigm = 10
cm =  mpl.colormaps['tab20']
topcol = [cm(i/(nbigm-1)) for i in range(nbigm)]
for ii,v in enumerate(res.index):
    vs = res.loc[v,:]
    indmax = np.argmax(np.abs(vs.iloc[:xlim]))
    if np.any(vs!=0) and np.where(vs!=0)[0][0] <= xlim:
        if cnt < len(topcol):
            col = topcol[cnt]
            label = v
        else:
            col = 'gray'
            label = None
        cnt += 1
    else:
        label = None
        col = 'gray'
    plt.plot(res.columns, res.loc[v,:], label = label, color = col)[0]
plt.legend(prop={'size':5}, loc = 'lower right')
ax = plt.gca()
ll, ul = ax.get_ylim()
plt.vlines(resdf['tau'][nll_ind], ll, ul, linestyle='--', color = 'gray')
plt.xscale('log')
plt.title("Means")

plt.subplot(2,1,2)
cnt = 0
#nbigz = max(x.shape[0] for x in df_means)
nbigz = 10
cm =  mpl.colormaps['tab20']
topcol = [cm(i/(nbigz-1)) for i in range(nbigz)]
for ii,v in enumerate(res0.index):
    vs = res0.loc[v,:]
    indmax = np.argmax(np.abs(vs.iloc[:xlim]))
    if np.any(vs!=0) and np.where(vs!=0)[0][0] <= xlim:
        if cnt < len(topcol):
            col = topcol[cnt]
            label = v
        else:
            col = 'gray'
            label = None
        cnt += 1
    else:
        label = None
        col = 'gray'
    plt.plot(res0.columns, res0.loc[v,:], label = label, color = col)[0]
plt.legend(prop={'size':5}, loc = 'lower right')
ax = plt.gca()
ll, ul = ax.get_ylim()
plt.vlines(resdf['tau'][nll_ind], ll, ul, linestyle='--', color = 'gray')
plt.xscale('log')
plt.title("Zeros")

plt.savefig('traj'+str(eu_only)+'.pdf')
plt.close()

###
#files = sorted(glob.glob('sim_out/'+simout_dir+'*'))
#dfs = {}
#for f in files:
#    dfs[f.split('/')[-1]] = pd.read_csv(f, index_col = 0)
#
#all_files = list(dfs.keys())
#cf = [x for x in all_files if 'comp' in x]
#comp_df = pd.concat([dfs[x] for x in cf])
