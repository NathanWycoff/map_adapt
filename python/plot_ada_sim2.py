#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  plot_ada_sim.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.18.2023

import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd

exec(open('python/sim_settings.py').read())

import glob
files = glob.glob('sim_out/'+sim+"_"+sparsity_type+"/*")
dfs = []
for f in files:
    dfs.append(pd.read_csv(f))
res = pd.concat(dfs)

#methods = sorted(list(set(res['Method'])))
methods = [x for x in model_colors.keys() if x in models2try]
ncomp = len(methods)

#if sparsity_type in ['group','hier2nd']:
#    if 'sbl_ada' in methods:
#        methods.remove('sbl_ada')

isjags = res['Method']=='jags'
isnan = res['yy-MSE']==1.0
res.loc[np.logical_and(isjags, isnan),'yy-MSE'] = np.nan
res.loc[np.logical_and(isjags, isnan),'Time'] = np.nan

cols = [model_colors[x] for x in methods]

#fig = plt.figure(figsize=[16,12]) # For my own inspection
if sim=='synthetic':
    nsc = len(Ns)*2
    set_inds = np.sort(list(set(res['Setting'])))
    nper = len(set_inds)//nsc

    #cols = ['red','blue','orange','green','cyan','purple']

    for targ in ['MSE','Time']:
        fig = plt.figure(figsize=[10,6]) # To fit in article
        #fig = plt.figure(figsize=[14,5]) # To fit in article
        for cc in range(nsc):
            # betamse
            
            col = cc//2
            row = cc%2
            plt.subplot(2,3,1+col+row*3)
            subset_inds = set_inds[(cc*nper):((cc+1)*nper)]

            for s_i in subset_inds:
                N = str(settings[int(s_i)][0])
                P = str(settings[int(s_i)][1])
                tp = "N="+N+"; Pu="+P
                if targ=='MSE':
                    subsets = [res.loc[(res['Method']==meth)*(res['Setting']==s_i),'beta-MSE'] for meth in methods]
                    title = r"$\beta$-MSE " + tp
                elif targ == 'Time':
                    subsets_mse = [res.loc[(res['Method']==meth)*(res['Setting']==s_i),'beta-MSE'] for meth in methods]
                    subsets = [res.loc[(res['Method']==meth)*(res['Setting']==s_i),'Time'] for meth in methods]
                    for si in range(len(subsets_mse)):
                        subsets[si][np.isnan(subsets_mse[si])] = np.nan
                    title = "Wall Time (s) " + tp
                else:
                    raise Exception

                subsets = [[x for x in ss if not np.isnan(x)] for ss in subsets]

                bplot = plt.boxplot(subsets, positions = [ncomp*s_i+i for i in range(len(methods))],patch_artist=True, medianprops=dict(color="black"), widths = 0.85)
                for patch, color in zip(bplot['boxes'], cols):
                    patch.set(color=color)
                    # change fill color
                    patch.set(facecolor = color)
                    #patch.set_facecolor(color)
                    #for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
                    #    plt.setp(bplot[item], color=color)
                    #plt.setp(box1["boxes"], facecolor=c2)
                    #plt.setp(box1["fliers"], markeredgecolor=c2)

                ax = plt.gca()
                hshift = 0.2
                ax.set_xticks([Ni*ncomp+(ncomp/2)-hshift for Ni in subset_inds])
                xlabs = [settings[int(x)][3].title() for x in subset_inds]
                if cc % 2 == 0:
                    for i in range(1):
                        xlabs[i] += "\n"+r"$\sigma$="+str(settings[i][4])
                ax.set_xticklabels(xlabs, fontdict = {'weight':'bold','size':9})
                #ax.set_xticklabels([int(x) for x in subset_inds])
                ll = ax.get_ylim()
                plt.vlines(subset_inds*ncomp-2*hshift, ll[0], ll[1], color = 'black')
                plt.yscale('log')
                if cc % 2 == 0:
                    if targ == 'MSE':
                        plt.title(title, fontdict={'weight':'bold'})
                    elif targ == 'Time':
                        plt.title(title, fontdict={'weight':'bold','size':10})
                    else:
                        raise Exception
                #if ii == 3:
                #    ls = []
                #    for mi,m in enumerate(methods):
                #        hB, = plt.plot([subset_inds[0]*ncomp,subset_inds[0]*ncomp],[ll[0],ll[1]], color = cols[mi])
                #        ls.append(hB)
                #    plt.legend(ls,methods)
                #    for l in ls:
                #        l.set_visible(False)

        plt.tight_layout()
        plt.savefig(sim+'_'+sparsity_type+"_sim_"+targ+".pdf")
        plt.close()

        linewidth = 0.15
        lines = []
        labels = []
        for i,m in enumerate(methods):
            lines.append(plt.hlines(0,i,i+linewidth, color = model_colors[m], linewidth=7))
            if m in nice_names.keys():
                title = nice_names[m]
            else:
                title = m
            labels.append(title)
            #plt.text(i+linewidth+0.05,0,title, font={'weight':'bold', 'size':9})
        fig = plt.figure(figsize=[10,0.5])
        #plt.legend(lines, labels, ncol = len(lines), prop = {'weight':'bold', 'size':10})
        #plt.legend(lines, labels, ncol = int(np.ceil(len(lines)/2)), prop = {'weight':'bold', 'size':15}, loc = 'center')
        plt.legend(lines, labels, ncol = len(lines), prop = {'weight':'bold', 'size':10}, loc = 'center')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("synth_legend_"+sparsity_type+".pdf")
        plt.close()


        ### Make legend in last 
        #for i in range(1):
        #    col = 3
        #    row = i
        #    plt.subplot(2,4,1+col+row*4)

        #    linewidth = 0.15
        #    lines = []
        #    labels = []
        #    for i,m in enumerate(methods):
        #        lines.append(plt.hlines(0,i,i+linewidth, color = model_colors[m], linewidth=7))
        #        if m in nice_names.keys():
        #            title = nice_names[m]
        #        else:
        #            title = m
        #        labels.append(title)
        #        #plt.text(i+linewidth+0.05,0,title, font={'weight':'bold', 'size':9})
        #    #fig = plt.figure(figsize=[12,1])
        #    #plt.legend(lines, labels, ncol = len(lines), prop = {'weight':'bold', 'size':10})
        #    #plt.legend(lines, labels, ncol = int(np.ceil(len(lines)/2)), prop = {'weight':'bold', 'size':15})
        #    plt.legend(lines, labels, loc = 'upper left')
        #    plt.axis('off')
        #    for i in range(len(lines)):
        #        lines[i].set_alpha(0)
        #    #plt.tight_layout()
        #    #plt.savefig("synth_legend.pdf")
        #    #plt.close()
else:
    for targ in ['MSE','Time']:
        #fig = plt.figure(figsize=[10,6]) # To fit in article
        #fig = plt.figure(figsize=[8,12]) # To fit in article
        fig = plt.figure(figsize=[12,6]) # To fit in article
        #fig = plt.figure(figsize=[14,5]) # To fit in article

        prob_names = list(set(res['Setting']))

        for pp in range(len(prob_names)):
            s_i = prob_names[pp]
            #plt.subplot(5,2,pp+1)
            plt.subplot(2,5,pp+1)

            #tp = "N="+N+"; P="+P
            if targ=='MSE':
                subsets = [res.loc[(res['Method']==meth)*(res['Setting']==s_i),'yy-MSE'] for meth in methods]
                title = s_i.title()
            elif targ == 'Time':
                subsets_mse = [res.loc[(res['Method']==meth)*(res['Setting']==s_i),'yy-MSE'] for meth in methods]
                subsets = [res.loc[(res['Method']==meth)*(res['Setting']==s_i),'Time'] for meth in methods]
                for si in range(len(subsets_mse)):
                    subsets[si][np.isnan(subsets_mse[si])] = np.nan
                title = "Wall Time (s) " + s_i.title()
            else:
                raise Exception

            subsets = [[x for x in ss if not np.isnan(x)] for ss in subsets]

            bplot = plt.boxplot(subsets, positions = [ncomp*pp+i for i in range(len(methods))],patch_artist=True, medianprops=dict(color="black"), widths = 0.85)
            for patch, color in zip(bplot['boxes'], cols):
                patch.set(color=color)
                patch.set(facecolor = color)

            ax = plt.gca()
            mini_names = {'sbl_ada':'LM-Indep','sbl_hier':'LM-Hier','OLS':'GLM','jags':'HorseShoe'}
            ax.set_xticklabels([mini_names[x] if x in mini_names.keys() else x for x in methods], font = {'weight':'bold','size':10})
            #ax.set_xticklabels(methods, font = {'weight':'bold','size':8})
            plt.xticks(rotation=45, ha='right')
            if s_i in class_problems:
                ylab = 'Misclassification Rate'
            else:
                ylab = 'MSE'
            plt.ylabel(ylab)
            #hshift = 0.2
            #ax.set_xticks([Ni*ncomp+(ncomp/2)-hshift for Ni in subset_inds])
            #xlabs = [settings[int(x)][3].title() for x in subset_inds]
            #if cc % 2 == 0:
            #    for i in range(3):
            #        xlabs[i] += "\n"+r"$\sigma$="+str(settings[i][4])
            #ax.set_xticklabels(xlabs, fontdict = {'weight':'bold','size':9})
            #ax.set_xticklabels([int(x) for x in subset_inds])
            #ll = ax.get_ylim()
            #plt.vlines(subset_inds*ncomp-2*hshift, ll[0], ll[1], color = 'black')
            #if ii==0 or ii==1:
            if s_i in ['abalone','bodyfat']:
                plt.yscale('log')
                if s_i == 'abalone':
                    ax.set_ylim(1,100)
            if s_i == 'triazines':
                ax.set_ylim(0,0.05)
            if s_i == 'pyrim':
                ax.set_ylim(0,0.1)
            if targ == 'MSE':
                plt.title(title, fontdict={'weight':'bold'})
            elif targ == 'Time':
                plt.title(title, fontdict={'weight':'bold','size':10})
            else:
                raise Exception
                    
        plt.tight_layout()
        plt.savefig(sim+'_'+sparsity_type+"_sim_"+targ+".pdf")
        plt.close()

### Table the LibSVM sim.
#pre = r"\\textbf{"
#post = "}"
#summ = res.groupby(['Setting','Method']).agg(Time=('Time',np.median), LowerQuantile=('yy-MSE', lambda x: np.quantile(x, 0.1)), Median=('yy-MSE', np.median), UpperQuantile=('yy-MSE', lambda x: np.quantile(x, 0.9)), Nonzero=('nonzero',np.median))
#summ.index.names = ['Dataset','Method']
#
## Scale all losses on bodyfat dataset to allow for consistent rounding.
#summ.loc[summ.index.get_level_values('Dataset')=='bodyfat',['LowerQuantile','Median','UpperQuantile']] *= 1000
#
#targ = ['Median']
#summ[targ] = summ[targ].round(4).astype(str)
#
#
#summ.index = summ.index.set_levels([nice_names[x] if x in nice_names.keys() else x for x in summ.index.get_level_values('Method')], level = 1, verify_integrity=False)
#
#new_ind = list(summ.index)
#for t in targ:
#    for s in list(set(res['Setting'])):
#        subset = summ.loc[s,t].astype(float)
#        winner = subset.index[np.where(subset==np.min(subset))[0]]
#        for w in winner:
#            aa = pre+str(summ.loc[summ.index==(s,w),t][0])+post
#            summ.loc[summ.index==(s,w),t] = aa
#        new_ind = [(x[0],pre+str(x[1])+post) if x[0]==s and x[1] in winner else x for x in new_ind]
#summ.index = pd.MultiIndex.from_tuples(new_ind)
#
#summ = summ.loc[~np.any(summ.isna(), axis = 1),:]
#summ['Nonzero'] = summ['Nonzero'].astype(int)
#
#summ.index = summ.index.set_names(['Dataset','Method'])
#
#import re
#with open('tables/'+sim+"_"+sparsity_type+'.txt', 'w') as f:
#    ss = summ.to_latex()
#    ss=re.sub(r"\\textbackslash \\textbackslash ",r"\\",ss)
#    ss=re.sub(r"\\{",r"{",ss)
#    ss=re.sub(r"\\}",r"}",ss)
#    f.write(ss)
