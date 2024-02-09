#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  plot_hcr.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 02.08.2024

import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd

exec(open('python/sim_settings.py').read())
exec(open('python/hcr_settings.py').read())

import glob
files = sorted(glob.glob('sim_out/'+simout_dir+'*'))
dfs = {}
for f in files:
    dfs[f.split('/')[-1]] = pd.read_csv(f, index_col = 0)

all_files = list(dfs.keys())

zf = [x for x in all_files if 'zinb' in x]
zinb_df = pd.concat([dfs[x] for x in zf])

mf = [x for x in all_files if 'betas_mean' in x]
mean_dfs = [dfs[x] for x in mf]

zf = [x for x in all_files if 'betas_zero' in x]
zero_dfs = [dfs[x] for x in zf]

#cf = [x for x in all_files if 'comp' in x]
#comp_df = pd.concat([dfs[x] for x in cf])

mean_dfs[-1]

zinb_df['nnz'] = [mean_dfs[i].shape[0]+zero_dfs[i].shape[0] for i in range(len(mean_dfs))]

zinb_df
