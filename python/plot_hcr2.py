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

with open("pickles/traj_hcr_"+str(eu_only)+'_'+expansion+'.pkl', 'rb') as f:
    df_means, df_zeros, resdf = pickle.load(f)

for i in range(len(df_means)):
    df_means[i].columns = [np.flip(tau_range)[i], 'name']
    df_means[i]['name'] = [x.replace('best_est','deaths') for x in df_means[i]['name']]
for i in range(len(df_zeros)):
    df_zeros[i].columns = [np.flip(tau_range)[i], 'name']
    df_zeros[i]['name'] = [x.replace('best_est','deaths') for x in df_zeros[i]['name']]

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

#medrad = 7
medrad = 15
#medrad = 3
for ii,v in enumerate(res.index):
    res.loc[v,:] =  medfilt(res.loc[v,:],medrad)
for ii,v in enumerate(res0.index):
    res0.loc[v,:] =  medfilt(res0.loc[v,:],medrad)

# To aid visualization keep everthing close.
#thresh = 0.75
thresh = 100
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

#xlim = 40 # for 50
xlim = 80
res = res.iloc[:,1:xlim]
res0 = res0.iloc[:,1:xlim]

#nll_ind = 35 # for 50.
nll_ind = 70
print("nll:")
print(resdf['nll'][nll_ind])

#xlim = 80
#xlim = 100
#xlim = 50

#fig = plt.figure(figsize=[5,2.5])
#fig = plt.figure(figsize=[1,1])

#plt.subplot(2,1,1)
#fig, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize=[5,3])
fig, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]}, figsize=[5,2])

vline = 34

cnt = 0
#nbigm = max(x.shape[0] for x in df_means)
#nbigm = 10
nbigm = 4
cm =  mpl.colormaps['tab20b']
topcol = [cm(i/(nbigm-1)) for i in range(nbigm)]
for ii,v in enumerate(res.index):
    vs = res.loc[v,:]
    indmax = np.argmax(np.abs(vs.iloc[1:xlim]))
    if np.any(vs!=0) and np.where(vs!=0)[0][0] <= xlim:
        if cnt < nbigm:
            col = topcol[cnt]
            label = v
        else:
            col = 'gray'
            label = None
        cnt += 1
    else:
        label = None
        col = 'gray'
    a0.plot(res.columns, res.loc[v,:], label = label, color = col)[0]
a0.legend(prop={'size':5}, loc = 'upper right', framealpha=1.)
ll, ul = a0.get_ylim()
a0.vlines(resdf['tau'][vline], ll, ul, linestyle='--', color = 'gray')
a0.set_xscale('log')
a0.set_title("Hurdle Model Coefficient Trajectory", fontsize=8)
a0.set_ylabel("Coefficent Estimates", fontsize = 8)
#a0.set_xticks([])
#a0.set_xticks([], minor=True)
a0.set_xlabel(r"$\tau$")

#plt.subplot(2,1,2)
nll = medfilt(resdf['nll'],medrad)
taus = resdf['tau']
a1.plot(taus[1:xlim], nll[1:xlim])
a1.set_xscale('log')
a1.set_title("Predictive NLL", fontsize = 8)
a1.set_xlabel(r"$\tau$")
a1.set_ylim([np.min(nll), np.max(nll[1:xlim])])
a1.tick_params(axis='both', which='major', labelsize=5)
a1.tick_params(axis='both', which='minor', labelsize=5)
a1.vlines(resdf['tau'][vline], np.min(nll), np.max(nll[1:xlim]), linestyle='--', color = 'gray')
a1.set_xlim([np.min(taus[1:xlim]), np.max(taus[1:xlim])])

plt.tight_layout()
plt.savefig('traj'+str(eu_only)+'_'+expansion+'.pdf')
plt.close()
