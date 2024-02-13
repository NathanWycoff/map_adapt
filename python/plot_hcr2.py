#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  plot_hcr2.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 02.09.2024

import pickle
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import medfilt
from adjustText import adjust_text
import glob

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

for ii,v in enumerate(res.index):
    #plt.plot(res.columns, medfilt(res.loc[v,:],7), label = v)
    res.loc[v,:] =  medfilt(res.loc[v,:],7)

# To aid visualization keep everthing close.
#res = np.maximum(-0.5, np.minimum(0.5, res))

#first_nz = np.where(res!=0)[1]
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

res = res.iloc[:,:90]
#res = res.iloc[:,:85]

nll_ind = 53
print("nll:")
print(resdf['nll'][nll_ind])

xlim = 80

fig = plt.figure(figsize=[5,2.5])
texts = []
objs = []
cnt = 0
topcol = ['red','blue','green','orange','purple','cyan']
for ii,v in enumerate(res.index):
    #if ii in first_inds:
    vs = res.loc[v,:]
    indmax = np.argmax(np.abs(vs.iloc[:xlim]))
    #if np.any(vs!=0) and np.where(vs!=0)[0][0] < xlim:
    #    fnz = np.where(vs!=0)[0][0]
    #    print(fnz)
    #    print(ii)
    #    print('--')
    #    xv = res.columns[fnz]
    #    #yv = np.abs(vs.iloc[indmax])*np.sign(vs.iloc[indmax])
    #    yv = 0.25*np.power(-1,cnt)
    #    cnt += 1
    #    texts.append(plt.text(xv, yv, v, font = {'size' : 8}))
    if np.any(vs!=0) and np.where(vs!=0)[0][0] < xlim:
        label = v
        col = topcol[cnt]
        cnt += 1
    else:
        label = None
        col = 'gray'
    objs.append(plt.plot(res.columns, res.loc[v,:], label = label, color = col)[0])
    #texts.append(plt.text(resdf['tau'][first_nz[ii]], np.abs(vs.iloc[indmax])*np.sign(vs.iloc[indmax]), 'x'))
plt.legend(prop={'size':6})
ax = plt.gca()
#labelLines(ax.get_lines(), zorder=2.5)
ax.set_ylim(-0.3, 0.3)
ll, ul = ax.get_ylim()
#adjust_text(texts, arrowprops=dict(arrowstyle="->", color='r', lw=0.5), objects = objs, only_move={"text": "y", "static": "y", "explode": "y", "pull": "y"},)
plt.vlines(resdf['tau'][nll_ind], ll, ul, linestyle='--', color = 'gray')
#plt.vlines(res.columns[xlim], ll, ul, linestyle='--', color = 'gray')
#plt.vlines(resdf['tau'][np.min(first_nz)], ll, ul, linestyle='--', color = 'gray')
plt.xscale('log')
plt.savefig('traj'+str(eu_only)+'.pdf')
plt.close()

##
files = sorted(glob.glob('sim_out/'+simout_dir+'*'))
dfs = {}
for f in files:
    dfs[f.split('/')[-1]] = pd.read_csv(f, index_col = 0)

all_files = list(dfs.keys())
cf = [x for x in all_files if 'comp' in x]
comp_df = pd.concat([dfs[x] for x in cf])
