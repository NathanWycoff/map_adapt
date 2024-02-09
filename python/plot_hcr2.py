#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  plot_hcr2.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 02.09.2024

import pickle
import matplotlib.pyplot as plt

exec(open('python/hcr_settings.py').read())

with open("pickles/traj_hcr_"+str(eu_only)+'.pdf', 'rb') as f:
    df_means, df_zeros, resdf = pickle.load(f)
