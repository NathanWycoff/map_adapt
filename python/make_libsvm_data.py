#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  make_libsvm_data.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 02.05.2024

from pathlib import Path
import urllib.request
from libsvm.svmutil import svm_read_problem
import bz2
import pandas as pd
from sklearn.datasets import load_svmlight_file
import numpy as np

exec(open('python/sim_settings.py').read())

Path(libsvm_dir).mkdir(parents=True, exist_ok=True)
Path(data_dir).mkdir(parents=True, exist_ok=True)

urls = {}
#urls['gisette'] = ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2', 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.t.bz2']
urls['abalone'] = ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/abalone']
urls['housing'] = ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/housing']
urls['bodyfat'] = ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/bodyfat']
urls['mpg'] = ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/mpg']
urls['mg'] = ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/mg']
urls['australian'] = ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/australian']
urls['diabetes'] = ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/diabetes']
urls['heart'] = ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/heart']
urls['covtype'] = ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.bz2']
urls['mushrooms'] = ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/mushrooms']
urls['phishing'] = ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/phishing']

#no_norm_y = ['abalone']

datasets = list(urls.keys())

for ds in datasets:
    dfs = []
    for ui,url in enumerate(urls[ds]):
        # Download bz2 file
        if url[-4:]=='.bz2':
            fname = libsvm_dir+ds+'_'+str(ui)+'.bz2'
        else:
            fname = libsvm_dir+ds+'_'+str(ui)
        urllib.request.urlretrieve(url, fname)

        dat = load_svmlight_file(fname)
        X = np.array(dat[0].todense())
        y = dat[1]
        df = pd.DataFrame(X)
        df.columns = ['X'+str(i) for i in range(X.shape[1])]
        df['y'] = y
        dfs.append(df)
    # Renormalize
    df = pd.concat(dfs, axis = 0)
    eps = 1e-6
    if ds in class_problems:
        df['y'] = (df['y']-min(df['y'])) / (max(df['y'])-min(df['y']))
    elif ds in reg_problems:
        df['y'] = (df['y'] - np.mean(df['y'])) / (np.std(df['y'])+eps)
    else:
        raise Exception("Unknown dataset type!")
    for i in range(df.shape[1]-1):
        df.iloc[:,i] = (df.iloc[:,i] - np.mean(df.iloc[:,i])) / (np.std(df.iloc[:,i])+eps)

    # Write to disk.

    outname = data_dir+ds+'.csv'
    df.to_csv(outname, index = False)

